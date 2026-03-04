/**
 * シールカウンター - ヤマザキ春のパンまつり 2026
 * 基本フロー: 撮影/ギャラリー → 1回スキャン → 確認・編集 → 決定
 */
;(function() {
  'use strict';

  // ====================================
  // 定数
  // ====================================
  var STORAGE_KEY = 'seal-counter-data';
  var GOAL_POINTS = 30;
  var POINT_VALUES = [0.5, 1, 1.5, 2, 2.5, 3];
  var MODEL_VERSION = '2.0.1';
  var MODEL_URL = '../models/seal-detector.onnx';
  var MODEL_CACHE_KEY = 'seal-detector-model';
  var YEAR_MODEL_URL = '../models/year-detector.onnx';
  var YEAR_MODEL_CACHE_KEY = 'year-detector-model';
  var YEAR_MODEL_VERSION = '1.0.0';
  var CONFIDENCE_HIGH = 0.5;

  // セッション永続化定数
  var SESSION_DB_NAME = 'seal-counter-session';
  var SESSION_STORE = 'session';
  var SESSION_UI_KEY = 'seal-counter-session-ui';
  var SESSION_MAX_AGE = 24 * 60 * 60 * 1000; // 24時間

  // セッション復元ガード
  var sessionRestoring = false;

  // ====================================
  // 状態管理
  // ====================================
  var state = {
    seals: [],
    totalPoints: 0,
    currentCaptureId: null,
    editingDetectionId: null,
    addingDetectionPos: null,
    highlightDetectionId: null,
    zoomedView: null,
    captureCanvasValid: false,
    modelLoaded: false,
    modelLoading: false,
    yearModelLoaded: false,
    scanning: false,
    cameraStream: null,
    pendingDetections: [],
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
    dom.cameraButtons = document.getElementById('camera-buttons');
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
    dom.scanCard = document.getElementById('scan-card');
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
        // YOLOモデル読み込み後、年号モデルも読み込む
        loadYearModel();
        break;
      case 'year-model-loaded':
        state.yearModelLoaded = true;
        console.log('年号検出モデル読み込み完了');
        break;
      case 'year-model-error':
        console.warn('年号検出モデル読み込み失敗（旧年度判定なしで動作）:', msg.message);
        break;
      case 'inference-result':
        handleInferenceResult(msg.detections, msg.elapsed);
        break;
      case 'error':
        console.error('Worker:', msg.message);
        state.modelLoading = false;
        state.scanning = false;
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
  // 年号検出モデル読み込み + キャッシュ
  // ====================================
  function loadYearModel() {
    getYearModelCache().then(function(cached) {
      if (cached && cached.version === YEAR_MODEL_VERSION) {
        worker.postMessage({ type: 'load-year-model', modelData: cached.data });
        return;
      }
      return fetch(YEAR_MODEL_URL).then(function(resp) {
        if (!resp.ok) throw new Error('年号モデルダウンロード失敗 (' + resp.status + ')');
        return resp.arrayBuffer();
      }).then(function(buf) {
        setYearModelCache(buf.slice(0), YEAR_MODEL_VERSION);
        worker.postMessage({ type: 'load-year-model', modelData: buf }, [buf]);
      });
    }).catch(function(e) {
      console.warn('年号モデル読み込みスキップ:', e.message);
    });
  }

  function getYearModelCache() {
    return new Promise(function(resolve) {
      try {
        var req = indexedDB.open('seal-counter-models', 1);
        req.onupgradeneeded = function(e) {
          var db = e.target.result;
          if (!db.objectStoreNames.contains('models')) {
            db.createObjectStore('models');
          }
        };
        req.onsuccess = function(e) {
          var db = e.target.result;
          var tx = db.transaction('models', 'readonly');
          var store = tx.objectStore('models');
          var g = store.get(YEAR_MODEL_CACHE_KEY);
          g.onsuccess = function() { resolve(g.result || null); };
          g.onerror = function() { resolve(null); };
        };
        req.onerror = function() { resolve(null); };
      } catch (e) { resolve(null); }
    });
  }

  function setYearModelCache(data, version) {
    return new Promise(function(resolve) {
      try {
        var req = indexedDB.open('seal-counter-models', 1);
        req.onupgradeneeded = function(e) {
          var db = e.target.result;
          if (!db.objectStoreNames.contains('models')) {
            db.createObjectStore('models');
          }
        };
        req.onsuccess = function(e) {
          var db = e.target.result;
          var tx = db.transaction('models', 'readwrite');
          tx.objectStore('models').put(
            { data: data, version: version, cachedAt: Date.now() },
            YEAR_MODEL_CACHE_KEY
          );
          tx.oncomplete = function() { resolve(); };
          tx.onerror = function() { resolve(); };
        };
        req.onerror = function() { resolve(); };
      } catch (e) { resolve(); }
    });
  }

  // ====================================
  // セッション永続化（ページ再読み込み耐性）
  // ====================================
  function openSessionDB() {
    return new Promise(function(resolve) {
      try {
        var req = indexedDB.open(SESSION_DB_NAME, 1);
        req.onupgradeneeded = function(e) {
          var db = e.target.result;
          if (!db.objectStoreNames.contains(SESSION_STORE)) {
            db.createObjectStore(SESSION_STORE);
          }
        };
        req.onsuccess = function(e) { resolve(e.target.result); };
        req.onerror = function() { resolve(null); };
      } catch (e) { resolve(null); }
    });
  }

  function saveSessionToIDB(viewState) {
    return new Promise(function(resolve) {
      captureImageBlob(function(blob) {
        openSessionDB().then(function(db) {
          if (!db) { resolve(); return; }
          try {
            var tx = db.transaction(SESSION_STORE, 'readwrite');
            var store = tx.objectStore(SESSION_STORE);
            store.put({
              viewState: viewState,
              pendingDetections: state.pendingDetections,
              savedAt: Date.now(),
            }, 'snapshot');
            if (blob) {
              store.put(blob, 'image');
            }
            tx.oncomplete = function() { resolve(); };
            tx.onerror = function() { resolve(); };
          } catch (e) { resolve(); }
        });
      });
    });
  }

  function loadSessionFromIDB() {
    return new Promise(function(resolve) {
      openSessionDB().then(function(db) {
        if (!db) { resolve(null); return; }
        try {
          var tx = db.transaction(SESSION_STORE, 'readonly');
          var store = tx.objectStore(SESSION_STORE);
          var snapReq = store.get('snapshot');
          var imgReq = store.get('image');
          tx.oncomplete = function() {
            var snapshot = snapReq.result;
            var imageBlob = imgReq.result || null;
            if (!snapshot) { resolve(null); return; }
            // 有効期限チェック
            if (Date.now() - snapshot.savedAt > SESSION_MAX_AGE) {
              clearSessionFromIDB();
              resolve(null);
              return;
            }
            resolve({
              viewState: snapshot.viewState,
              pendingDetections: snapshot.pendingDetections || [],
              imageBlob: imageBlob,
            });
          };
          tx.onerror = function() { resolve(null); };
        } catch (e) { resolve(null); }
      });
    });
  }

  function clearSessionFromIDB() {
    return new Promise(function(resolve) {
      openSessionDB().then(function(db) {
        if (!db) { resolve(); return; }
        try {
          var tx = db.transaction(SESSION_STORE, 'readwrite');
          var store = tx.objectStore(SESSION_STORE);
          store.delete('snapshot');
          store.delete('image');
          tx.oncomplete = function() { resolve(); };
          tx.onerror = function() { resolve(); };
        } catch (e) { resolve(); }
      });
    });
  }

  function saveSessionUI(viewState) {
    try { localStorage.setItem(SESSION_UI_KEY, viewState); } catch (e) {}
  }

  function clearSessionUI() {
    try { localStorage.removeItem(SESSION_UI_KEY); } catch (e) {}
  }

  function captureImageBlob(callback) {
    // captureCanvas が有効な場合（元画像をボックスなしで保存）
    var cc = dom.captureCanvas;
    if (state.captureCanvasValid && cc && cc.width > 1) {
      cc.toBlob(function(b) { callback(b); }, 'image/jpeg', 0.85);
      return;
    }
    // detectOverlay にバックアップ画像がある場合（detected状態、旧形式フォールバック）
    var overlay = dom.detectOverlay;
    if (overlay && !overlay.classList.contains('hidden') && overlay.width > 1) {
      overlay.toBlob(function(b) { callback(b); }, 'image/jpeg', 0.85);
      return;
    }
    // capturedImage が表示中の場合（captured状態）
    var img = dom.capturedImage;
    if (img && img.src && !img.classList.contains('hidden') && img.naturalWidth > 0) {
      var MAX_DIM = 1280;
      var w = img.naturalWidth;
      var h = img.naturalHeight;
      if (w > MAX_DIM || h > MAX_DIM) {
        var scale = MAX_DIM / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      var c = document.createElement('canvas');
      c.width = w;
      c.height = h;
      c.getContext('2d').drawImage(img, 0, 0, w, h);
      c.toBlob(function(b) { callback(b); }, 'image/jpeg', 0.85);
      return;
    }
    callback(null);
  }

  function saveSession(viewState) {
    if (sessionRestoring) return;
    saveSessionUI(viewState);
    saveSessionToIDB(viewState);
  }

  function clearSession() {
    clearSessionUI();
    clearSessionFromIDB();
  }

  function restoreSession(data) {
    sessionRestoring = true;
    var url = URL.createObjectURL(data.imageBlob);

    if (data.viewState === 'detected' && data.pendingDetections.length > 0) {
      // 検出結果あり: 元画像をcaptureCanvasに復元してからボックス再描画
      state.pendingDetections = data.pendingDetections;
      var overlayImg = new Image();
      overlayImg.onload = function() {
        URL.revokeObjectURL(url);
        // captureCanvas に元画像を復元
        dom.captureCanvas.width = overlayImg.width;
        dom.captureCanvas.height = overlayImg.height;
        dom.captureCanvas.getContext('2d').drawImage(overlayImg, 0, 0);
        state.captureCanvasValid = true;
        // drawDetections: captureCanvas → detectOverlay + ボックス描画
        drawDetections();

        // UI状態を検出結果表示に
        dom.cameraPlaceholder.classList.add('hidden');
        dom.cameraPreview.classList.add('hidden');
        dom.capturedImage.classList.add('hidden');
        dom.cameraButtons.classList.add('hidden');
        if (dom.detectResults) dom.detectResults.classList.remove('hidden');
        // Step 3: detectOverlay がタップを受ける
        dom.detectOverlay.style.pointerEvents = 'auto';

        renderDetectionList();
        updateDetectStats();
        updateStepIndicator(3);
        sessionRestoring = false;
      };
      overlayImg.onerror = function() {
        URL.revokeObjectURL(url);
        clearSession();
        sessionRestoring = false;
      };
      overlayImg.src = url;
    } else {
      // captured状態: 撮影画像を復元
      dom.cameraPlaceholder.classList.add('hidden');
      dom.cameraPreview.classList.add('hidden');
      dom.capturedImage.classList.remove('hidden');
      dom.cameraButtons.classList.add('hidden');
      if (dom.scanCard) dom.scanCard.classList.remove('hidden');

      dom.capturedImage.onload = function() {
        dom.capturedImage.onload = null;
        updateStepIndicator(2);
        sessionRestoring = false;
      };
      dom.capturedImage.onerror = function() {
        dom.capturedImage.onerror = null;
        URL.revokeObjectURL(url);
        clearSession();
        resetCameraView();
        sessionRestoring = false;
      };
      dom.capturedImage.src = url;
      state.currentCaptureId = Date.now().toString(36);
    }
  }

  // ====================================
  // カメラ停止
  // ====================================
  function stopCamera() {
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

    state.captureCanvasValid = true;
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
    saveSession('detected');
    updateStepIndicator(3);
  }

  // ====================================
  // 検出結果表示
  // ====================================
  function showDetectionResults(elapsed) {
    if (!dom.detectResults) return;
    dom.detectResults.classList.remove('hidden');
    if (dom.scanCard) dom.scanCard.classList.add('hidden');
    drawDetections();
    // captureCanvas は redraw 用に保持（confirmDetections/resetCameraView で解放）
    // blob URL 画像も解放（detectOverlay が表示を担うため不要）
    if (dom.capturedImage.src && dom.capturedImage.src.startsWith('blob:')) {
      URL.revokeObjectURL(dom.capturedImage.src);
    }
    dom.capturedImage.src = '';
    // Step 3: detectOverlay がタップを受ける
    dom.detectOverlay.style.pointerEvents = 'auto';
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

  function redrawDetections() {
    var canvas = dom.detectOverlay;
    var src = dom.captureCanvas;
    if (!canvas || !src || src.width <= 1) return;

    var zv = state.zoomedView;
    var ctx;
    if (zv) {
      // ズーム: コンテナARに合わせたキャンバスサイズで切り出し描画
      var rect = canvas.getBoundingClientRect();
      var cAR = rect.width / rect.height;
      canvas.width = 1280;
      canvas.height = Math.round(1280 / cAR);
      ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(src, zv.sx, zv.sy, zv.viewW, zv.viewH, 0, 0, canvas.width, canvas.height);
      // 以降の描画を元画像座標系で行えるよう変換設定
      var scX = canvas.width / zv.viewW, scY = canvas.height / zv.viewH;
      ctx.setTransform(scX, 0, 0, scY, -zv.sx * scX, -zv.sy * scY);
    } else {
      canvas.width = src.width;
      canvas.height = src.height;
      ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(src, 0, 0);
    }

    var hlId = state.highlightDetectionId;
    state.pendingDetections.forEach(function(det) {
      var b = det.bbox;
      var x = b[0], y = b[1], w = b[2] - b[0], h = b[3] - b[1];
      var isOld = det.classId === 6;
      var isLow = det.confidence < CONFIDENCE_HIGH;
      var isHL = (hlId != null && det.id === hlId);

      if (isHL) {
        ctx.fillStyle = 'rgba(229, 0, 90, 0.25)';
        ctx.fillRect(x, y, w, h);
      }

      if (isHL) { ctx.strokeStyle = '#E5005A'; ctx.setLineDash([]); ctx.lineWidth = 4; }
      else if (isOld) { ctx.strokeStyle = '#f97316'; ctx.setLineDash([6, 3]); ctx.lineWidth = 2; }
      else if (isLow) { ctx.strokeStyle = '#eab308'; ctx.setLineDash([3, 3]); ctx.lineWidth = 2; }
      else { ctx.strokeStyle = '#22c55e'; ctx.setLineDash([]); ctx.lineWidth = 2; }

      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      var label = isOld ? '旧年度?' : (isLow ? '要確認' : det.points + '点');
      var fontSize = Math.max(12, Math.min(16, w / 4));
      ctx.font = 'bold ' + fontSize + 'px sans-serif';
      var tw = ctx.measureText(label).width;
      ctx.fillStyle = isHL ? '#E5005A' : (isOld ? '#f97316' : (isLow ? '#eab308' : '#22c55e'));
      ctx.fillRect(x, y - fontSize - 4, tw + 8, fontSize + 4);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x + 4, y - 4);
    });

    // 変換をリセット
    ctx.setTransform(1, 0, 0, 1, 0, 0);
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
        '<button class="detect-point-btn inline-flex items-center justify-center w-8 h-8 rounded-full text-white text-xs font-bold ' + badge + '" data-detect-id="' + det.id + '">' + label + '</button>' +
        '<span class="flex-grow text-gray-400 text-xs">' + conf + '</span>' +
        '<button class="detect-toggle-btn ' + tColor + ' text-xs font-bold" data-detect-id="' + det.id + '">' + toggle + '</button>' +
        '<button class="detect-delete-btn text-gray-300 text-xs px-1" data-detect-id="' + det.id + '">' + '\u2715' + '</button>' +
      '</div>';
    }).join('');
  }

  function buildPointBreakdown(items, pointKey) {
    var counts = {};
    POINT_VALUES.forEach(function(v) { counts[v] = 0; });
    items.forEach(function(item) { var p = item[pointKey]; if (counts[p] != null) counts[p]++; });
    var parts = [];
    POINT_VALUES.forEach(function(v) { if (counts[v] > 0) parts.push(v + '点\u00d7' + counts[v]); });
    return parts.join('  ');
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
    var bdEl = document.getElementById('detect-breakdown');
    if (bdEl) {
      var text = valid.length > 0 ? buildPointBreakdown(valid, 'points') : '';
      bdEl.textContent = text;
    }
  }

  function confirmDetections() {
    var accepted = state.pendingDetections.filter(function(d) { return d.accepted; });
    var captureId = Date.now().toString(36);
    accepted.forEach(function(det) { addSeal(det.points, captureId, 'scan'); });
    state.pendingDetections = [];
    resetCameraView();
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
  // UI ヘルパー
  // ====================================
  function updateModelStatus(status, text) {
    if (!dom.modelStatus) return;
    var dot = dom.modelStatus.querySelector('.status-dot');
    var textEl = dom.modelStatus.querySelector('.status-text');
    if (textEl) {
      textEl.textContent = text;
    } else {
      dom.modelStatus.textContent = text;
    }

    var base = 'flex items-center text-2xs ';
    var dotBase = 'status-dot ';
    // ヘッダーがピンク背景なので白系トーンで表示
    switch (status) {
      case 'loading':
        dom.modelStatus.style.color = 'rgba(255,255,255,0.7)';
        if (dot) dot.className = dotBase + 'is-loading';
        break;
      case 'ready':
        dom.modelStatus.style.color = 'rgba(255,255,255,0.9)';
        if (dot) dot.className = dotBase;
        break;
      case 'error':
        dom.modelStatus.style.color = 'rgba(255,200,200,0.9)';
        if (dot) dot.className = dotBase;
        break;
      default:
        dom.modelStatus.style.color = 'rgba(255,255,255,0.5)';
        if (dot) dot.className = dotBase;
        break;
    }
  }

  function updateScanUI(scanning) {
    if (dom.scanBtn) {
      dom.scanBtn.disabled = scanning || !state.modelLoaded;
      dom.scanBtn.textContent = scanning ? '認識中...' : 'スキャンする';
    }
    if (dom.scanStatus) dom.scanStatus.classList.toggle('hidden', !scanning);
  }

  function updateStepIndicator(step) {
    for (var i = 1; i <= 3; i++) {
      var el = document.getElementById('ui-step-' + i);
      if (!el) continue;
      el.classList.remove('active', 'completed');
      if (i < step) el.classList.add('completed');
      else if (i === step) el.classList.add('active');
    }
    for (var j = 1; j <= 2; j++) {
      var conn = document.getElementById('ui-conn-' + j);
      if (!conn) continue;
      if (j < step) conn.classList.add('done');
      else conn.classList.remove('done');
    }
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
    var sbEl = document.getElementById('seal-breakdown');
    if (sbEl) {
      sbEl.textContent = seals.length > 0 ? buildPointBreakdown(seals, 'point') : '';
    }
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
    resetCameraView();
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
    dom.cameraButtons.classList.add('hidden');
    if (dom.scanCard) dom.scanCard.classList.remove('hidden');
    state.currentCaptureId = Date.now().toString(36);

    dom.capturedImage.onload = function() {
      dom.capturedImage.onload = null;
      saveSession('captured');
    };
    dom.capturedImage.src = src;
    updateStepIndicator(2);
  }

  function resetCameraView() {
    dom.cameraPlaceholder.classList.remove('hidden');
    dom.capturedImage.classList.add('hidden');
    // Object URL 解放
    if (dom.capturedImage.src && dom.capturedImage.src.startsWith('blob:')) {
      URL.revokeObjectURL(dom.capturedImage.src);
    }
    dom.capturedImage.src = '';
    dom.cameraPreview.classList.add('hidden');
    dom.cameraButtons.classList.remove('hidden');
    state.currentCaptureId = null;
    if (dom.detectResults) dom.detectResults.classList.add('hidden');
    if (dom.detectOverlay) {
      dom.detectOverlay.classList.add('hidden');
      dom.detectOverlay.style.pointerEvents = '';
    }
    if (dom.scanCard) dom.scanCard.classList.add('hidden');
    // captureCanvas メモリ解放
    dom.captureCanvas.width = 1;
    dom.captureCanvas.height = 1;
    state.captureCanvasValid = false;
    clearSession();
    updateStepIndicator(1);
  }

  function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) return;
    // createObjectURL: base64変換を避けてメモリ節約（iOS Safari クラッシュ防止）
    var url = URL.createObjectURL(file);
    showCapturedImage(url);
  }

  // ====================================
  // Step 3: object-contain 座標変換ヘルパー
  // ====================================
  function getContainLayout() {
    var el = dom.detectOverlay;
    if (!el) return null;
    var rect = el.getBoundingClientRect();
    var cw = el.width, ch = el.height;
    if (cw <= 1 || rect.width <= 0) return null;
    var sx = rect.width / cw, sy = rect.height / ch;
    var scale = Math.min(sx, sy);
    var contentW = cw * scale, contentH = ch * scale;
    var offsetX = (rect.width - contentW) / 2;
    var offsetY = (rect.height - contentH) / 2;
    return { scale: scale, offsetX: offsetX, offsetY: offsetY, rect: rect, cw: cw, ch: ch };
  }

  function clientToCanvas(clientX, clientY) {
    var L = getContainLayout();
    if (!L) return null;
    return {
      x: (clientX - L.rect.left - L.offsetX) / L.scale,
      y: (clientY - L.rect.top - L.offsetY) / L.scale,
    };
  }

  // ====================================
  // Step 3: キャンバスレベルズーム
  // ====================================
  function zoomToDetection(detId) {
    var det = state.pendingDetections.find(function(d) { return d.id === detId; });
    if (!det) return;
    var b = det.bbox;
    var bw = b[2] - b[0], bh = b[3] - b[1];
    var src = dom.captureCanvas;
    if (!src || src.width <= 1) return;
    // ボックスが表示領域の約35%を占めるズーム倍率
    var zoom = Math.min(src.width / (bw * 2.8), src.height / (bh * 2.8));
    zoom = Math.max(2, Math.min(6, zoom));
    var cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
    setZoomedView(cx, cy, zoom);
  }

  function zoomToPosition(canvasX, canvasY) {
    setZoomedView(canvasX, canvasY, 3);
  }

  function setZoomedView(cx, cy, zoom) {
    var src = dom.captureCanvas;
    var canvas = dom.detectOverlay;
    if (!src || src.width <= 1 || !canvas) return;
    // コンテナのアスペクト比に合わせたビューポートを計算
    var rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    var containerAR = rect.width / rect.height;
    // 元画像の短辺ベースでズーム量を決定
    var baseSize = Math.min(src.width, src.height) / zoom;
    var viewW, viewH;
    if (containerAR >= 1) {
      viewH = baseSize;
      viewW = viewH * containerAR;
    } else {
      viewW = baseSize;
      viewH = viewW / containerAR;
    }
    // 元画像の範囲に収まるよう調整
    if (viewW > src.width) { viewW = src.width; viewH = viewW / containerAR; }
    if (viewH > src.height) { viewH = src.height; viewW = viewH * containerAR; }
    var sx = cx - viewW / 2, sy = cy - viewH / 2;
    sx = Math.max(0, Math.min(src.width - viewW, sx));
    sy = Math.max(0, Math.min(src.height - viewH, sy));
    state.zoomedView = { sx: sx, sy: sy, viewW: viewW, viewH: viewH };
    redrawDetections();
  }

  function resetDetectionZoom() {
    if (!state.zoomedView) return;
    state.zoomedView = null;
    redrawDetections();
  }

  // ====================================
  // Step 3: 検出オーバーレイタップ（手動追加）
  // ====================================
  function handleDetectOverlayTap(e) {
    if (!dom.detectResults || dom.detectResults.classList.contains('hidden')) return;
    e.preventDefault();
    var coords = clientToCanvas(e.clientX, e.clientY);
    if (!coords) return;
    var canvasX = coords.x, canvasY = coords.y;

    // ズーム中のタップは座標を元画像空間に変換
    var zv = state.zoomedView;
    if (zv) {
      var cvsW = dom.detectOverlay.width, cvsH = dom.detectOverlay.height;
      canvasX = zv.sx + canvasX * (zv.viewW / cvsW);
      canvasY = zv.sy + canvasY * (zv.viewH / cvsH);
    }

    // 画像外タップを無視
    var cw = dom.captureCanvas.width, ch = dom.captureCanvas.height;
    if (canvasX < 0 || canvasY < 0 || canvasX > cw || canvasY > ch) return;

    // 既存ボックスとの当たり判定
    var hitDet = null;
    for (var i = state.pendingDetections.length - 1; i >= 0; i--) {
      var d = state.pendingDetections[i];
      var b = d.bbox;
      if (canvasX >= b[0] && canvasX <= b[2] && canvasY >= b[1] && canvasY <= b[3]) {
        hitDet = d;
        break;
      }
    }

    if (hitDet) {
      state.editingDetectionId = hitDet.id;
      state.highlightDetectionId = hitDet.id;
      zoomToDetection(hitDet.id);
      showPointModal(hitDet.points);
    } else {
      state.addingDetectionPos = { x: canvasX, y: canvasY };
      zoomToPosition(canvasX, canvasY);
      showPointModal();
    }
  }

  function addManualDetection(pos, points) {
    var estimatedSize = 60;
    var maxId = state.pendingDetections.reduce(function(m, d) { return Math.max(m, d.id); }, -1);
    var newDet = {
      id: maxId + 1,
      bbox: [pos.x - estimatedSize / 2, pos.y - estimatedSize / 2,
             pos.x + estimatedSize / 2, pos.y + estimatedSize / 2],
      classId: POINT_VALUES.indexOf(points),
      className: points + '点',
      confidence: 1.0,
      points: points,
      accepted: true,
    };
    state.pendingDetections.push(newDet);
    renderDetectionList();
    redrawDetections();
    updateDetectStats();
    saveSession('detected');
  }

  // ====================================
  // モーダル
  // ====================================
  function showPointModal(currentPoints) {
    // 現在の点数をハイライト表示
    document.querySelectorAll('.modal-point-btn').forEach(function(btn) {
      var p = parseFloat(btn.dataset.point);
      if (currentPoints != null && p === currentPoints) {
        btn.classList.add('modal-point-selected');
      } else {
        btn.classList.remove('modal-point-selected');
      }
    });
    dom.pointModal.classList.remove('hidden');
  }
  function hidePointModal() {
    dom.pointModal.classList.add('hidden');
    state.editingDetectionId = null;
    state.addingDetectionPos = null;
    // ハイライト＋ズーム解除（まとめて1回再描画）
    var needRedraw = state.highlightDetectionId != null || state.zoomedView != null;
    state.highlightDetectionId = null;
    state.zoomedView = null;
    if (needRedraw) redrawDetections();
  }
  function showResetModal() { dom.resetModal.classList.remove('hidden'); }
  function hideResetModal() { dom.resetModal.classList.add('hidden'); }

  // ====================================
  // イベントバインド
  // ====================================
  function bindEvents() {
    document.getElementById('btn-camera').addEventListener('click', function() { dom.cameraInput.click(); });
    document.getElementById('btn-gallery').addEventListener('click', function() { dom.fileInput.click(); });
    dom.fileInput.addEventListener('change', function(e) { if (e.target.files[0]) { handleFileSelect(e.target.files[0]); e.target.value = ''; } });
    dom.cameraInput.addEventListener('change', function(e) { if (e.target.files[0]) { handleFileSelect(e.target.files[0]); e.target.value = ''; } });
    document.getElementById('btn-retake').addEventListener('click', resetCameraView);
    document.getElementById('btn-retake-step3').addEventListener('click', resetCameraView);

    document.querySelectorAll('.point-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var p = parseFloat(this.dataset.point);
        if (!isNaN(p)) addSeal(p, null, 'manual');
      });
    });

    document.querySelectorAll('.modal-point-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var p = parseFloat(this.dataset.point);
        if (isNaN(p)) return;

        if (state.editingDetectionId != null) {
          var det = state.pendingDetections.find(function(d) { return d.id === state.editingDetectionId; });
          if (det) {
            if (det.points === p) {
              // 同じ点数を再タップ → 不採用にして削除
              det.accepted = false;
              renderDetectionList();
              redrawDetections();
              updateDetectStats();
              saveSession('detected');
            } else {
              // 検出の点数変更
              det.classId = POINT_VALUES.indexOf(p);
              det.points = p;
              det.className = p + '点';
              renderDetectionList();
              redrawDetections();
              updateDetectStats();
              saveSession('detected');
            }
          }
          hidePointModal();
        } else if (state.addingDetectionPos) {
          // Step 3手動追加
          addManualDetection(state.addingDetectionPos, p);
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
      runScan();
    });
    if (dom.confirmDetections) dom.confirmDetections.addEventListener('click', confirmDetections);

    if (dom.detectionList) {
      dom.detectionList.addEventListener('click', function(e) {
        var detectId;
        // 点数変更ボタン
        var pb = e.target.closest('.detect-point-btn');
        if (pb) {
          detectId = parseInt(pb.dataset.detectId, 10);
          var det = state.pendingDetections.find(function(d) { return d.id === detectId; });
          state.editingDetectionId = detectId;
          // 対応するボックスをハイライト＋ズーム
          state.highlightDetectionId = detectId;
          zoomToDetection(detectId);
          showPointModal(det ? det.points : null);
          return;
        }
        // 削除ボタン
        var db = e.target.closest('.detect-delete-btn');
        if (db) {
          detectId = parseInt(db.dataset.detectId, 10);
          state.pendingDetections = state.pendingDetections.filter(function(d) { return d.id !== detectId; });
          renderDetectionList();
          redrawDetections();
          updateDetectStats();
          saveSession('detected');
          return;
        }
        // 採用/除外トグル
        var tb = e.target.closest('.detect-toggle-btn');
        if (!tb) return;
        var det = state.pendingDetections.find(function(d) { return d.id === parseInt(tb.dataset.detectId, 10); });
        if (det) { det.accepted = !det.accepted; renderDetectionList(); updateDetectStats(); saveSession('detected'); }
      });
    }
    // Step 3: 検出オーバーレイタップで手動追加
    if (dom.detectOverlay) {
      dom.detectOverlay.addEventListener('pointerdown', handleDetectOverlayTap);
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

    // セッション復元: localStorage同期チェック → IndexedDB非同期読み取り
    var savedUI = null;
    try { savedUI = localStorage.getItem(SESSION_UI_KEY); } catch (e) {}
    if (savedUI) {
      // フラッシュ防止: プレースホルダーを即座に非表示
      dom.cameraPlaceholder.classList.add('hidden');
      loadSessionFromIDB().then(function(data) {
        if (data && data.imageBlob) {
          restoreSession(data);
        } else {
          // IndexedDBにデータがない場合はUIを初期状態に戻す
          dom.cameraPlaceholder.classList.remove('hidden');
          clearSessionUI();
        }
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
