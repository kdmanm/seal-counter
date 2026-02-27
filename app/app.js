/**
 * シールカウンター - ヤマザキ春のパンまつり 2026
 * Phase 1: 半自動スキャン + 手動入力
 * Phase 2: 連続映像処理
 */
;(function() {
  'use strict';

  // ====================================
  // 定数
  // ====================================
  var STORAGE_KEY = 'seal-counter-data';
  var GOAL_POINTS = 30;
  var POINT_VALUES = [0.5, 1, 1.5, 2, 2.5, 3];
  var MODEL_VERSION = '1.0.0';
  var MODEL_URL = '../models/seal-detector.onnx';
  var MODEL_CACHE_KEY = 'seal-detector-model';
  var CONFIDENCE_HIGH = 0.5;

  // Phase 2: 連続スキャン定数
  var CONTINUOUS_THROTTLE_MS = 200;   // 5fps目標
  var TRACK_IOU_THRESHOLD = 0.3;      // トラック照合IoU閾値
  var TRACK_MAX_LOST = 10;            // 消失許容フレーム数
  var TRACK_CONFIRM_FRAMES = 15;      // 自動確定に必要な連続検出フレーム数 (~3秒@5fps)
  var trackNextId = 1;

  // ====================================
  // 状態管理
  // ====================================
  var state = {
    seals: [],
    totalPoints: 0,
    currentCaptureId: null,
    markerPendingPos: null,
    modelLoaded: false,
    modelLoading: false,
    scanning: false,
    cameraStream: null,
    pendingDetections: [],
    // Phase 2
    scanMode: 'single',         // 'single' | 'continuous'
    continuousRunning: false,
    inferenceInFlight: false,
    trackedObjects: [],
    confirmedTrackIds: {},
    rafId: null,
    lastInferenceTime: 0,
  };

  // ====================================
  // DOM参照
  // ====================================
  var dom = {};

  function cacheDom() {
    dom.totalPoints = document.getElementById('total-points');
    dom.platesEarned = document.getElementById('plates-earned');
    dom.progressBar = document.getElementById('progress-bar');
    dom.progressMessage = document.getElementById('progress-message');
    dom.sealList = document.getElementById('seal-list');
    dom.sealEmpty = document.getElementById('seal-empty');
    dom.sealCount = document.getElementById('seal-count');
    dom.cameraPreview = document.getElementById('camera-preview');
    dom.captureCanvas = document.getElementById('capture-canvas');
    dom.capturedImage = document.getElementById('captured-image');
    dom.cameraPlaceholder = document.getElementById('camera-placeholder');
    dom.markerOverlay = document.getElementById('marker-overlay');
    dom.cameraButtons = document.getElementById('camera-buttons');
    dom.postCaptureActions = document.getElementById('post-capture-actions');
    dom.pointModal = document.getElementById('point-modal');
    dom.resetModal = document.getElementById('reset-modal');
    dom.fileInput = document.getElementById('file-input');
    dom.cameraInput = document.getElementById('camera-input');
    dom.scanBtn = document.getElementById('btn-scan');
    dom.scanStatus = document.getElementById('scan-status');
    dom.modelStatus = document.getElementById('model-status');
    dom.detectOverlay = document.getElementById('detect-overlay');
    dom.detectResults = document.getElementById('detect-results');
    dom.confirmDetections = document.getElementById('btn-confirm-detections');
    dom.detectionList = document.getElementById('detection-list');
    dom.shootingGuide = document.getElementById('shooting-guide');
    // Phase 2
    dom.btnModeSingle = document.getElementById('btn-mode-single');
    dom.btnModeContinuous = document.getElementById('btn-mode-continuous');
    dom.continuousIndicator = document.getElementById('continuous-indicator');
    dom.continuousStats = document.getElementById('continuous-stats');
    dom.continuousDetected = document.getElementById('continuous-detected');
    dom.continuousConfirmed = document.getElementById('continuous-confirmed');
  }

  // ====================================
  // Web Worker
  // ====================================
  var worker = null;

  function initWorker() {
    worker = new Worker('worker.js');
    worker.onmessage = handleWorkerMessage;
    worker.onerror = function(e) {
      console.error('Worker error:', e);
      updateModelStatus('error', 'Worker エラー');
    };
  }

  function handleWorkerMessage(e) {
    var msg = e.data;
    switch (msg.type) {
      case 'model-loaded':
        state.modelLoaded = true;
        state.modelLoading = false;
        updateModelStatus('ready', 'モデル準備完了');
        if (dom.scanBtn) dom.scanBtn.disabled = false;
        break;
      case 'inference-result':
        if (state.scanMode === 'continuous' && state.continuousRunning) {
          state.inferenceInFlight = false;
          updateTracker(msg.detections);
          drawContinuousOverlay();
          updateContinuousStats();
        } else {
          handleInferenceResult(msg.detections, msg.elapsed);
        }
        break;
      case 'error':
        console.error('Worker:', msg.message);
        state.modelLoading = false;
        state.scanning = false;
        state.inferenceInFlight = false;
        updateModelStatus('error', msg.message);
        updateScanUI(false);
        break;
    }
  }

  // ====================================
  // モデル読み込み + キャッシュ
  // ====================================
  function loadModel() {
    if (state.modelLoaded || state.modelLoading) return;
    state.modelLoading = true;
    updateModelStatus('loading', 'モデルを読み込み中...');

    getModelCache().then(function(cached) {
      if (cached && cached.version === MODEL_VERSION) {
        worker.postMessage({ type: 'load-model', modelData: cached.data });
        return;
      }
      return fetch(MODEL_URL).then(function(resp) {
        if (!resp.ok) throw new Error('ダウンロード失敗 (' + resp.status + ')');
        return resp.arrayBuffer();
      }).then(function(buf) {
        // キャッシュ用にコピーを保存（transfer後は元bufが使用不可になるため）
        setModelCache(buf.slice(0), MODEL_VERSION);
        worker.postMessage({ type: 'load-model', modelData: buf }, [buf]);
      });
    }).catch(function(e) {
      state.modelLoading = false;
      console.warn('モデル読み込みスキップ:', e.message);
      updateModelStatus('unavailable', '手動入力モードで動作中');
    });
  }

  function getModelCache() {
    return new Promise(function(resolve) {
      try {
        var req = indexedDB.open('seal-counter-models', 1);
        req.onupgradeneeded = function(e) {
          e.target.result.createObjectStore('models');
        };
        req.onsuccess = function(e) {
          var db = e.target.result;
          var tx = db.transaction('models', 'readonly');
          var store = tx.objectStore('models');
          var g = store.get(MODEL_CACHE_KEY);
          g.onsuccess = function() { resolve(g.result || null); };
          g.onerror = function() { resolve(null); };
        };
        req.onerror = function() { resolve(null); };
      } catch (e) { resolve(null); }
    });
  }

  function setModelCache(data, version) {
    return new Promise(function(resolve) {
      try {
        var req = indexedDB.open('seal-counter-models', 1);
        req.onupgradeneeded = function(e) {
          e.target.result.createObjectStore('models');
        };
        req.onsuccess = function(e) {
          var db = e.target.result;
          var tx = db.transaction('models', 'readwrite');
          tx.objectStore('models').put(
            { data: data, version: version, cachedAt: Date.now() },
            MODEL_CACHE_KEY
          );
          tx.oncomplete = function() { resolve(); };
          tx.onerror = function() { resolve(); };
        };
        req.onerror = function() { resolve(); };
      } catch (e) { resolve(); }
    });
  }

  // ====================================
  // カメラ（ライブプレビュー）
  // ====================================
  function startCamera() {
    if (state.cameraStream) return;
    navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
    }).then(function(stream) {
      state.cameraStream = stream;
      dom.cameraPreview.srcObject = stream;
      dom.cameraPreview.classList.remove('hidden');
      dom.cameraPlaceholder.classList.add('hidden');
      dom.cameraPreview.play();
      if (dom.shootingGuide && !localStorage.getItem('seal-counter-guide-dismissed')) {
        dom.shootingGuide.classList.remove('hidden');
      }
    }).catch(function(e) {
      console.warn('カメラ起動失敗:', e.message);
    });
  }

  function stopCamera() {
    if (state.continuousRunning) stopContinuousScan();
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(function(t) { t.stop(); });
      state.cameraStream = null;
      dom.cameraPreview.srcObject = null;
    }
  }

  // ====================================
  // スキャン実行
  // ====================================
  function runScan() {
    if (!state.modelLoaded || state.scanning) return;
    state.scanning = true;
    updateScanUI(true);

    var canvas = dom.captureCanvas;
    var video = dom.cameraPreview;
    var ctx;

    var MAX_DIM = 1280;
    if (video.videoWidth > 0 && !video.classList.contains('hidden')) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);
    } else {
      var img = dom.capturedImage;
      if (img.src && !img.classList.contains('hidden')) {
        // 大きな写真はリサイズ（iOS Safari メモリクラッシュ防止）
        var w = img.naturalWidth;
        var h = img.naturalHeight;
        if (w > MAX_DIM || h > MAX_DIM) {
          var scale = MAX_DIM / Math.max(w, h);
          w = Math.round(w * scale);
          h = Math.round(h * scale);
        }
        canvas.width = w;
        canvas.height = h;
        ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
      } else {
        state.scanning = false;
        updateScanUI(false);
        return;
      }
    }

    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    worker.postMessage({
      type: 'inference',
      imageData: imageData,
      width: canvas.width,
      height: canvas.height,
    });
  }

  function handleInferenceResult(detections, elapsed) {
    state.scanning = false;
    state.pendingDetections = detections.map(function(d, i) {
      return {
        id: i, bbox: d.bbox, classId: d.classId, className: d.className,
        confidence: d.confidence, points: d.points,
        accepted: d.classId < 6,
      };
    });
    updateScanUI(false);
    showDetectionResults(elapsed);
    if (detections.length > 0) {
      if (navigator.vibrate) navigator.vibrate(50);
      playBeep();
    }
  }

  // ====================================
  // 検出結果表示
  // ====================================
  function showDetectionResults(elapsed) {
    if (!dom.detectResults) return;
    dom.detectResults.classList.remove('hidden');
    drawDetections();
    // detectOverlay に描画済みなので captureCanvas を解放（iOS メモリ節約）
    dom.captureCanvas.width = 1;
    dom.captureCanvas.height = 1;
    // blob URL 画像も解放（detectOverlay が表示を担うため不要）
    if (dom.capturedImage.src && dom.capturedImage.src.startsWith('blob:')) {
      URL.revokeObjectURL(dom.capturedImage.src);
    }
    dom.capturedImage.src = '';
    renderDetectionList();
    updateDetectStats(elapsed);
  }

  function drawDetections() {
    var canvas = dom.detectOverlay;
    if (!canvas) return;
    var src = dom.captureCanvas;
    canvas.width = src.width;
    canvas.height = src.height;
    canvas.classList.remove('hidden');
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(src, 0, 0);

    state.pendingDetections.forEach(function(det) {
      var b = det.bbox;
      var x = b[0], y = b[1], w = b[2] - b[0], h = b[3] - b[1];
      var isOld = det.classId === 6;
      var isLow = det.confidence < CONFIDENCE_HIGH;

      if (isOld) { ctx.strokeStyle = '#f97316'; ctx.setLineDash([6, 3]); }
      else if (isLow) { ctx.strokeStyle = '#eab308'; ctx.setLineDash([3, 3]); }
      else { ctx.strokeStyle = '#22c55e'; ctx.setLineDash([]); }

      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      var label = isOld ? '旧年度?' : (isLow ? '要確認' : det.points + '点');
      var fontSize = Math.max(12, Math.min(16, w / 4));
      ctx.font = 'bold ' + fontSize + 'px sans-serif';
      var tw = ctx.measureText(label).width;
      ctx.fillStyle = isOld ? '#f97316' : (isLow ? '#eab308' : '#22c55e');
      ctx.fillRect(x, y - fontSize - 4, tw + 8, fontSize + 4);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + 4, y - 4);
    });
  }

  function renderDetectionList() {
    if (!dom.detectionList) return;
    dom.detectionList.innerHTML = state.pendingDetections.map(function(det) {
      var isOld = det.classId === 6;
      var badge = det.accepted ? 'bg-green-500' : (isOld ? 'bg-orange-400' : 'bg-yellow-400');
      var label = isOld ? '旧' : det.points;
      var conf = Math.round(det.confidence * 100) + '%';
      var toggle = det.accepted ? '除外' : '追加';
      var tColor = det.accepted ? 'text-red-400' : 'text-green-500';
      return '<div class="flex items-center gap-2 py-1.5 px-2 bg-white rounded-lg text-sm">' +
        '<span class="inline-flex items-center justify-center w-8 h-8 rounded-full text-white text-xs font-bold ' + badge + '">' + label + '</span>' +
        '<span class="flex-grow text-gray-400 text-xs">' + conf + '</span>' +
        '<button class="detect-toggle-btn ' + tColor + ' text-xs font-bold" data-detect-id="' + det.id + '">' + toggle + '</button>' +
      '</div>';
    }).join('');
  }

  function updateDetectStats(elapsed) {
    var valid = state.pendingDetections.filter(function(d) { return d.accepted; });
    var pts = valid.reduce(function(s, d) { return s + d.points; }, 0);
    pts = Math.round(pts * 10) / 10;
    var el = document.getElementById('detect-stats');
    if (el) {
      el.textContent = valid.length + '枚採用 / ' + pts + '点' +
        (elapsed ? ' (' + elapsed + 'ms)' : '');
    }
  }

  function confirmDetections() {
    var accepted = state.pendingDetections.filter(function(d) { return d.accepted; });
    var captureId = Date.now().toString(36);
    accepted.forEach(function(det) { addSeal(det.points, captureId, 'scan'); });
    state.pendingDetections = [];
    if (dom.detectResults) dom.detectResults.classList.add('hidden');
    if (dom.detectOverlay) {
      dom.detectOverlay.classList.add('hidden');
      // キャンバスメモリ解放
      dom.detectOverlay.width = 1;
      dom.detectOverlay.height = 1;
    }
  }

  // ====================================
  // 効果音
  // ====================================
  function playBeep() {
    try {
      var ac = new (window.AudioContext || window.webkitAudioContext)();
      var osc = ac.createOscillator();
      var gain = ac.createGain();
      osc.connect(gain);
      gain.connect(ac.destination);
      osc.frequency.value = 880;
      gain.gain.value = 0.1;
      osc.start();
      osc.stop(ac.currentTime + 0.1);
    } catch (e) {}
  }

  // ====================================
  // Phase 2: IoU トラッカー
  // ====================================
  function computeIoU(a, b) {
    var x1 = Math.max(a[0], b[0]);
    var y1 = Math.max(a[1], b[1]);
    var x2 = Math.min(a[2], b[2]);
    var y2 = Math.min(a[3], b[3]);
    var inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (inter === 0) return 0;
    var areaA = (a[2] - a[0]) * (a[3] - a[1]);
    var areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (areaA + areaB - inter);
  }

  function matchDetections(tracks, detections) {
    var costs = [];
    for (var ti = 0; ti < tracks.length; ti++) {
      for (var di = 0; di < detections.length; di++) {
        var iou = computeIoU(tracks[ti].bbox, detections[di].bbox);
        if (iou >= TRACK_IOU_THRESHOLD) {
          costs.push({ ti: ti, di: di, iou: iou });
        }
      }
    }
    costs.sort(function(a, b) { return b.iou - a.iou; });

    var matchedT = {};
    var matchedD = {};
    var assignments = [];
    for (var i = 0; i < costs.length; i++) {
      var c = costs[i];
      if (!matchedT[c.ti] && !matchedD[c.di]) {
        assignments.push(c);
        matchedT[c.ti] = true;
        matchedD[c.di] = true;
      }
    }
    var unmatchedT = [];
    for (var t = 0; t < tracks.length; t++) {
      if (!matchedT[t]) unmatchedT.push(t);
    }
    var unmatchedD = [];
    for (var d = 0; d < detections.length; d++) {
      if (!matchedD[d]) unmatchedD.push(d);
    }
    return { assignments: assignments, unmatchedT: unmatchedT, unmatchedD: unmatchedD };
  }

  function updateTracker(detections) {
    var result = matchDetections(state.trackedObjects, detections);

    // マッチした既存トラックを更新
    for (var i = 0; i < result.assignments.length; i++) {
      var a = result.assignments[i];
      var track = state.trackedObjects[a.ti];
      var det = detections[a.di];
      track.bbox = det.bbox;
      track.classId = det.classId;
      track.className = det.className;
      track.confidence = det.confidence;
      track.points = det.points;
      track.seenFrames += 1;
      track.lostFrames = 0;

      // 自動確定チェック
      if (track.seenFrames >= TRACK_CONFIRM_FRAMES
          && !track.confirmed
          && track.classId < 6
          && track.confidence >= CONFIDENCE_HIGH
          && !state.confirmedTrackIds[track.id]) {
        autoConfirmTrack(track);
      }
    }

    // 未マッチのトラック: lostFrames++
    for (var j = 0; j < result.unmatchedT.length; j++) {
      state.trackedObjects[result.unmatchedT[j]].lostFrames += 1;
    }

    // 長期消失トラックを除去
    state.trackedObjects = state.trackedObjects.filter(function(t) {
      return t.lostFrames < TRACK_MAX_LOST;
    });

    // 未マッチの検出: 新規トラック作成
    for (var k = 0; k < result.unmatchedD.length; k++) {
      var nd = detections[result.unmatchedD[k]];
      state.trackedObjects.push({
        id: trackNextId++,
        bbox: nd.bbox,
        classId: nd.classId,
        className: nd.className,
        confidence: nd.confidence,
        points: nd.points,
        seenFrames: 1,
        lostFrames: 0,
        confirmed: false,
      });
    }
  }

  function autoConfirmTrack(track) {
    track.confirmed = true;
    state.confirmedTrackIds[track.id] = true;
    var captureId = 'continuous_' + Date.now().toString(36);
    addSeal(track.points, captureId, 'scan');
    if (navigator.vibrate) navigator.vibrate([50, 30, 50]);
    playBeep();
  }

  // ====================================
  // Phase 2: 連続スキャン
  // ====================================
  function startContinuousScan() {
    if (state.continuousRunning) return;
    if (!state.modelLoaded) return;

    // カメラが未起動なら起動
    if (!state.cameraStream) {
      navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      }).then(function(stream) {
        state.cameraStream = stream;
        dom.cameraPreview.srcObject = stream;
        dom.cameraPreview.classList.remove('hidden');
        dom.cameraPlaceholder.classList.add('hidden');
        dom.cameraPreview.play();
        beginContinuousLoop();
      }).catch(function(e) {
        console.warn('カメラ起動失敗:', e.message);
      });
    } else {
      beginContinuousLoop();
    }
  }

  function beginContinuousLoop() {
    state.continuousRunning = true;
    state.inferenceInFlight = false;
    state.trackedObjects = [];
    state.confirmedTrackIds = {};
    state.lastInferenceTime = 0;
    trackNextId = 1;

    // UI更新
    dom.continuousIndicator.classList.remove('hidden');
    dom.continuousStats.classList.remove('hidden');
    dom.detectOverlay.classList.remove('hidden');
    dom.cameraButtons.classList.add('hidden');
    if (dom.detectResults) dom.detectResults.classList.add('hidden');

    dom.scanBtn.textContent = '連続スキャン停止';
    dom.scanBtn.classList.add('continuous-stop');
    dom.scanBtn.disabled = false;

    updateContinuousStats();
    state.rafId = requestAnimationFrame(continuousFrame);
  }

  function stopContinuousScan() {
    state.continuousRunning = false;
    if (state.rafId) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }

    // UI復帰
    dom.continuousIndicator.classList.add('hidden');
    dom.continuousStats.classList.add('hidden');
    dom.detectOverlay.classList.add('hidden');
    dom.cameraButtons.classList.remove('hidden');

    dom.scanBtn.textContent = state.scanMode === 'continuous' ? '連続スキャン開始' : 'スキャンする';
    dom.scanBtn.classList.remove('continuous-stop');
    dom.scanBtn.disabled = !state.modelLoaded;

    // オーバーレイクリア
    var canvas = dom.detectOverlay;
    if (canvas) {
      var ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    state.trackedObjects = [];
    state.confirmedTrackIds = {};
    state.inferenceInFlight = false;
  }

  function continuousFrame(timestamp) {
    if (!state.continuousRunning) return;

    // 次のフレーム予約
    state.rafId = requestAnimationFrame(continuousFrame);

    // スロットリング
    if (timestamp - state.lastInferenceTime < CONTINUOUS_THROTTLE_MS) return;
    // Worker がまだ処理中なら待つ
    if (state.inferenceInFlight) return;

    var video = dom.cameraPreview;
    if (!video || video.videoWidth === 0) return;

    var canvas = dom.captureCanvas;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(video, 0, 0);

    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    state.inferenceInFlight = true;
    state.lastInferenceTime = timestamp;

    worker.postMessage({
      type: 'inference',
      imageData: imageData,
      width: canvas.width,
      height: canvas.height,
    });
  }

  function drawContinuousOverlay() {
    var canvas = dom.detectOverlay;
    if (!canvas) return;
    var video = dom.cameraPreview;
    if (!video || video.videoWidth === 0) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.classList.remove('hidden');
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    state.trackedObjects.forEach(function(track) {
      var b = track.bbox;
      var x = b[0], y = b[1], w = b[2] - b[0], h = b[3] - b[1];
      var isOld = track.classId === 6;
      var alpha = track.lostFrames > 0 ? Math.max(0.2, 1 - track.lostFrames / TRACK_MAX_LOST) : 1;

      ctx.globalAlpha = alpha;

      if (isOld) {
        ctx.strokeStyle = '#f97316';
        ctx.setLineDash([6, 3]);
        ctx.lineWidth = 2;
      } else if (track.confirmed) {
        ctx.strokeStyle = '#22c55e';
        ctx.setLineDash([]);
        ctx.lineWidth = 3;
      } else {
        ctx.strokeStyle = '#22c55e';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 2;
      }

      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      // ラベル
      var fontSize = Math.max(12, Math.min(16, w / 4));
      ctx.font = 'bold ' + fontSize + 'px sans-serif';
      var label;
      if (isOld) {
        label = '旧';
      } else if (track.confirmed) {
        label = track.points + '点 ✓';
      } else {
        label = track.points + '点 ' + track.seenFrames + '/' + TRACK_CONFIRM_FRAMES;
      }
      var tw = ctx.measureText(label).width;
      var bgColor = isOld ? '#f97316' : (track.confirmed ? '#16a34a' : '#059669');
      ctx.fillStyle = bgColor;
      ctx.fillRect(x, y - fontSize - 4, tw + 8, fontSize + 4);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + 4, y - 4);

      ctx.globalAlpha = 1;
    });
  }

  function updateContinuousStats() {
    if (!dom.continuousDetected) return;
    var active = state.trackedObjects.filter(function(t) { return t.lostFrames === 0; });
    var confirmed = Object.keys(state.confirmedTrackIds).length;
    dom.continuousDetected.textContent = active.length;
    dom.continuousConfirmed.textContent = confirmed;
  }

  // ====================================
  // Phase 2: モード切替
  // ====================================
  function setScanMode(mode) {
    if (mode === state.scanMode) return;
    if (state.continuousRunning) stopContinuousScan();

    state.scanMode = mode;

    // トグルUI
    if (mode === 'continuous') {
      dom.btnModeContinuous.className = 'flex-1 py-2 text-sm font-bold text-center scan-mode-active';
      dom.btnModeSingle.className = 'flex-1 py-2 text-sm font-bold text-center scan-mode-inactive';
      dom.scanBtn.textContent = '連続スキャン開始';
    } else {
      dom.btnModeSingle.className = 'flex-1 py-2 text-sm font-bold text-center scan-mode-active';
      dom.btnModeContinuous.className = 'flex-1 py-2 text-sm font-bold text-center scan-mode-inactive';
      dom.scanBtn.textContent = 'スキャンする';
    }
  }

  // ====================================
  // UI ヘルパー
  // ====================================
  function updateModelStatus(status, text) {
    if (!dom.modelStatus) return;
    dom.modelStatus.textContent = text;
    var base = 'text-xs ';
    switch (status) {
      case 'loading': dom.modelStatus.className = base + 'text-yellow-500'; break;
      case 'ready':   dom.modelStatus.className = base + 'text-green-500'; break;
      case 'error':   dom.modelStatus.className = base + 'text-red-500'; break;
      default:        dom.modelStatus.className = base + 'text-gray-400'; break;
    }
  }

  function updateScanUI(scanning) {
    if (dom.scanBtn) {
      dom.scanBtn.disabled = scanning || !state.modelLoaded;
      if (state.scanMode === 'continuous') {
        dom.scanBtn.textContent = state.continuousRunning ? '停止する' : '連続スキャン開始';
      } else {
        dom.scanBtn.textContent = scanning ? '認識中...' : 'スキャンする';
      }
    }
    if (dom.scanStatus) dom.scanStatus.classList.toggle('hidden', !scanning);
  }

  // ====================================
  // 永続化
  // ====================================
  function saveState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        seals: state.seals, savedAt: Date.now()
      }));
    } catch (e) { console.warn('保存に失敗:', e); }
  }

  function loadState() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      var data = JSON.parse(raw);
      if (data && Array.isArray(data.seals)) {
        state.seals = data.seals;
        recalcTotal();
      }
    } catch (e) { console.warn('読み込みに失敗:', e); }
  }

  // ====================================
  // ポイント計算
  // ====================================
  function recalcTotal() {
    state.totalPoints = state.seals.reduce(function(s, seal) { return s + seal.point; }, 0);
    state.totalPoints = Math.round(state.totalPoints * 10) / 10;
  }

  // ====================================
  // UI更新
  // ====================================
  function updateProgress() {
    var total = state.totalPoints;
    var plates = Math.floor(total / GOAL_POINTS);
    var remaining = GOAL_POINTS - (total % GOAL_POINTS);
    var pct = ((total % GOAL_POINTS) / GOAL_POINTS) * 100;

    dom.totalPoints.textContent = total;
    dom.platesEarned.textContent = plates;

    if (total >= GOAL_POINTS) {
      dom.progressBar.style.width = '100%';
      dom.progressBar.classList.add('bg-green-500');
      dom.progressBar.classList.remove('bg-rose-500');
    } else {
      dom.progressBar.style.width = pct + '%';
      dom.progressBar.classList.remove('bg-green-500');
      dom.progressBar.classList.add('bg-rose-500');
    }

    if (total === 0) {
      dom.progressMessage.textContent = 'あと 30 点';
    } else if (total >= GOAL_POINTS && remaining === GOAL_POINTS) {
      dom.progressMessage.textContent = plates + ' 枚達成！おめでとうございます！';
    } else if (total >= GOAL_POINTS) {
      dom.progressMessage.textContent = plates + ' 枚達成！次のお皿まであと ' + Math.round(remaining * 10) / 10 + ' 点';
    } else {
      dom.progressMessage.textContent = 'あと ' + Math.round(remaining * 10) / 10 + ' 点';
    }

    if (total >= GOAL_POINTS) {
      var section = document.getElementById('progress-section');
      section.classList.add('goal-reached');
      setTimeout(function() { section.classList.remove('goal-reached'); }, 2000);
    }
  }

  function renderSealList() {
    var seals = state.seals;
    dom.sealCount.textContent = seals.length + ' 枚';
    if (seals.length === 0) {
      dom.sealList.innerHTML = '';
      dom.sealEmpty.classList.remove('hidden');
      return;
    }
    dom.sealEmpty.classList.add('hidden');
    var rev = seals.slice().reverse();
    dom.sealList.innerHTML = rev.map(function(seal, idx) {
      var ri = seals.length - 1 - idx;
      var t = formatTime(seal.timestamp);
      var icon = seal.source === 'scan' ? '<span class="text-xs">AI</span>' : '';
      return '<div class="seal-entry' + (idx === 0 && seal._isNew ? ' seal-entry-new' : '') + '">' +
        '<div class="seal-point-badge">' + seal.point + '</div>' +
        '<div class="seal-info">' + icon + ' ' + t + '</div>' +
        '<button class="seal-delete-btn" data-index="' + ri + '" title="削除">' +
          '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">' +
            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />' +
          '</svg></button></div>';
    }).join('');
    seals.forEach(function(s) { delete s._isNew; });
  }

  function formatTime(ts) {
    var d = new Date(ts);
    return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  }

  function updateUI() { updateProgress(); renderSealList(); }

  // ====================================
  // シール操作
  // ====================================
  function addSeal(point, captureId, source) {
    state.seals.push({
      id: Date.now() + '_' + Math.random().toString(36).slice(2, 6),
      point: point, timestamp: Date.now(),
      captureId: captureId || null, source: source || 'manual', _isNew: true,
    });
    recalcTotal(); saveState(); updateUI();
    if (navigator.vibrate) navigator.vibrate(30);
  }

  function removeSeal(index) {
    if (index < 0 || index >= state.seals.length) return;
    state.seals.splice(index, 1);
    recalcTotal(); saveState(); updateUI();
  }

  function resetAll() {
    state.seals = []; state.totalPoints = 0;
    state.currentCaptureId = null; state.pendingDetections = [];
    // Phase 2: トラッキング状態クリア
    state.trackedObjects = []; state.confirmedTrackIds = {};
    trackNextId = 1;
    clearMarkers(); resetCameraView();
    saveState(); updateUI();
  }

  // ====================================
  // カメラ / 画像取り込み
  // ====================================
  function showCapturedImage(src) {
    stopCamera();
    dom.cameraPlaceholder.classList.add('hidden');
    dom.cameraPreview.classList.add('hidden');
    dom.capturedImage.classList.remove('hidden');
    dom.capturedImage.src = src;
    dom.markerOverlay.classList.remove('hidden');
    dom.cameraButtons.classList.add('hidden');
    dom.postCaptureActions.classList.remove('hidden');
    dom.postCaptureActions.classList.add('flex');
    state.currentCaptureId = Date.now().toString(36);
  }

  function resetCameraView() {
    if (state.continuousRunning) stopContinuousScan();
    dom.cameraPlaceholder.classList.remove('hidden');
    dom.capturedImage.classList.add('hidden');
    // Object URL 解放
    if (dom.capturedImage.src && dom.capturedImage.src.startsWith('blob:')) {
      URL.revokeObjectURL(dom.capturedImage.src);
    }
    dom.capturedImage.src = '';
    dom.cameraPreview.classList.add('hidden');
    dom.markerOverlay.classList.add('hidden');
    dom.cameraButtons.classList.remove('hidden');
    dom.postCaptureActions.classList.add('hidden');
    dom.postCaptureActions.classList.remove('flex');
    state.currentCaptureId = null;
    clearMarkers();
    if (dom.detectResults) dom.detectResults.classList.add('hidden');
    if (dom.detectOverlay) dom.detectOverlay.classList.add('hidden');
  }

  function clearMarkers() { dom.markerOverlay.innerHTML = ''; }

  function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) return;
    // createObjectURL: base64変換を避けてメモリ節約（iOS Safari クラッシュ防止）
    var url = URL.createObjectURL(file);
    showCapturedImage(url);
  }

  // ====================================
  // マーカーオーバーレイ
  // ====================================
  function handleOverlayTap(e) {
    e.preventDefault();
    var rect = dom.markerOverlay.getBoundingClientRect();
    var touch = e.touches ? e.touches[0] : e;
    var xPct = ((touch.clientX - rect.left) / rect.width) * 100;
    var yPct = ((touch.clientY - rect.top) / rect.height) * 100;
    state.markerPendingPos = { x: xPct, y: yPct };
    showPointModal();
  }

  function addMarker(xPct, yPct, point) {
    var m = document.createElement('div');
    m.className = 'tap-marker';
    m.style.left = xPct + '%';
    m.style.top = yPct + '%';
    m.setAttribute('data-point', point + '点');
    dom.markerOverlay.appendChild(m);
  }

  // ====================================
  // モーダル
  // ====================================
  function showPointModal() { dom.pointModal.classList.remove('hidden'); }
  function hidePointModal() { dom.pointModal.classList.add('hidden'); state.markerPendingPos = null; }
  function showResetModal() { dom.resetModal.classList.remove('hidden'); }
  function hideResetModal() { dom.resetModal.classList.add('hidden'); }

  // ====================================
  // イベントバインド
  // ====================================
  function bindEvents() {
    document.getElementById('btn-camera').addEventListener('click', function() { dom.cameraInput.click(); });
    var btnLive = document.getElementById('btn-live-camera');
    if (btnLive) btnLive.addEventListener('click', startCamera);
    document.getElementById('btn-gallery').addEventListener('click', function() { dom.fileInput.click(); });
    dom.fileInput.addEventListener('change', function(e) { if (e.target.files[0]) { handleFileSelect(e.target.files[0]); e.target.value = ''; } });
    dom.cameraInput.addEventListener('change', function(e) { if (e.target.files[0]) { handleFileSelect(e.target.files[0]); e.target.value = ''; } });
    document.getElementById('btn-retake').addEventListener('click', resetCameraView);
    document.getElementById('btn-confirm-marks').addEventListener('click', resetCameraView);
    dom.markerOverlay.addEventListener('pointerdown', handleOverlayTap);

    document.querySelectorAll('.point-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var p = parseFloat(this.dataset.point);
        if (!isNaN(p)) addSeal(p, null, 'manual');
      });
    });

    document.querySelectorAll('.modal-point-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var p = parseFloat(this.dataset.point);
        if (!isNaN(p) && state.markerPendingPos) {
          addMarker(state.markerPendingPos.x, state.markerPendingPos.y, p);
          addSeal(p, state.currentCaptureId, 'manual');
          hidePointModal();
        }
      });
    });

    document.getElementById('btn-modal-cancel').addEventListener('click', hidePointModal);
    dom.pointModal.addEventListener('click', function(e) { if (e.target === dom.pointModal) hidePointModal(); });

    document.getElementById('btn-reset').addEventListener('click', showResetModal);
    document.getElementById('btn-reset-cancel').addEventListener('click', hideResetModal);
    document.getElementById('btn-reset-confirm').addEventListener('click', function() { hideResetModal(); resetAll(); });
    dom.resetModal.addEventListener('click', function(e) { if (e.target === dom.resetModal) hideResetModal(); });

    dom.sealList.addEventListener('click', function(e) {
      var btn = e.target.closest('.seal-delete-btn');
      if (btn) removeSeal(parseInt(btn.dataset.index, 10));
    });

    // 推論 UI
    if (dom.scanBtn) dom.scanBtn.addEventListener('click', function() {
      if (state.scanMode === 'continuous') {
        if (state.continuousRunning) { stopContinuousScan(); }
        else { startContinuousScan(); }
      } else {
        runScan();
      }
    });
    if (dom.confirmDetections) dom.confirmDetections.addEventListener('click', confirmDetections);

    // Phase 2: モード切替
    if (dom.btnModeSingle) dom.btnModeSingle.addEventListener('click', function() { setScanMode('single'); });
    if (dom.btnModeContinuous) dom.btnModeContinuous.addEventListener('click', function() { setScanMode('continuous'); });
    if (dom.detectionList) {
      dom.detectionList.addEventListener('click', function(e) {
        var tb = e.target.closest('.detect-toggle-btn');
        if (!tb) return;
        var det = state.pendingDetections.find(function(d) { return d.id === parseInt(tb.dataset.detectId, 10); });
        if (det) { det.accepted = !det.accepted; renderDetectionList(); updateDetectStats(); }
      });
    }
    var gc = document.getElementById('guide-dismiss');
    if (gc) gc.addEventListener('click', function() {
      if (dom.shootingGuide) dom.shootingGuide.classList.add('hidden');
      localStorage.setItem('seal-counter-guide-dismissed', '1');
    });
  }

  // ====================================
  // 初期化
  // ====================================
  function init() {
    cacheDom(); loadState(); updateUI(); bindEvents();
    initWorker(); loadModel();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
