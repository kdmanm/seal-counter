/**
 * シールカウンター - Web Worker
 * ONNX Runtime Web による YOLOv8-nano 推論実行
 */

// ONNX Runtime Web CDN
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js');

// 定数
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const NUM_CLASSES = 7;
const PADDING_COLOR = 114; // RGB(114,114,114) — Ultralytics デフォルト

const CLASS_NAMES = [
  'seal_05', 'seal_1', 'seal_15', 'seal_2',
  'seal_25', 'seal_3', 'seal_old'
];

const POINT_VALUES = [0.5, 1, 1.5, 2, 2.5, 3, 0];

let session = null;

/**
 * モデルを読み込む。
 */
async function loadModel(modelData) {
  try {
    // WebGL を優先、フォールバックで WASM
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

    const options = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    };

    if (modelData instanceof ArrayBuffer) {
      session = await ort.InferenceSession.create(modelData, options);
    } else {
      session = await ort.InferenceSession.create(modelData, options);
    }

    // 入出力情報
    const inputNames = session.inputNames;
    const outputNames = session.outputNames;

    self.postMessage({
      type: 'model-loaded',
      inputNames: inputNames,
      outputNames: outputNames,
    });
  } catch (e) {
    self.postMessage({
      type: 'error',
      message: 'モデル読み込みに失敗: ' + e.message,
    });
  }
}

/**
 * 画像データを前処理してテンソルに変換する。
 * letterbox リサイズ + 正規化 + CHW 転置
 */
function preprocess(imageData, srcWidth, srcHeight) {
  const scale = Math.min(INPUT_SIZE / srcWidth, INPUT_SIZE / srcHeight);
  const newW = Math.round(srcWidth * scale);
  const newH = Math.round(srcHeight * scale);
  const padX = Math.round((INPUT_SIZE - newW) / 2);
  const padY = Math.round((INPUT_SIZE - newH) / 2);

  // Float32Array [1, 3, 640, 640]
  const inputTensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

  // padding で初期化
  const padVal = PADDING_COLOR / 255.0;
  inputTensor.fill(padVal);

  // リサイズ + 配置（最近傍補間）
  const srcData = imageData.data; // RGBA Uint8ClampedArray

  for (let y = 0; y < newH; y++) {
    const srcY = Math.min(Math.floor(y / scale), srcHeight - 1);
    for (let x = 0; x < newW; x++) {
      const srcX = Math.min(Math.floor(x / scale), srcWidth - 1);
      const srcIdx = (srcY * srcWidth + srcX) * 4;

      const dstX = x + padX;
      const dstY = y + padY;

      if (dstX >= 0 && dstX < INPUT_SIZE && dstY >= 0 && dstY < INPUT_SIZE) {
        const pixelIdx = dstY * INPUT_SIZE + dstX;

        // CHW 形式 + 正規化 (0-255 → 0.0-1.0)
        inputTensor[0 * INPUT_SIZE * INPUT_SIZE + pixelIdx] = srcData[srcIdx] / 255.0;     // R
        inputTensor[1 * INPUT_SIZE * INPUT_SIZE + pixelIdx] = srcData[srcIdx + 1] / 255.0; // G
        inputTensor[2 * INPUT_SIZE * INPUT_SIZE + pixelIdx] = srcData[srcIdx + 2] / 255.0; // B
      }
    }
  }

  return {
    tensor: inputTensor,
    scale: scale,
    padX: padX,
    padY: padY,
  };
}

/**
 * NMS (Non-Maximum Suppression)
 */
function nms(boxes, scores, iouThreshold) {
  const indices = [];
  const order = scores
    .map((s, i) => [s, i])
    .sort((a, b) => b[0] - a[0])
    .map(pair => pair[1]);

  const suppressed = new Set();

  for (const i of order) {
    if (suppressed.has(i)) continue;
    indices.push(i);

    for (const j of order) {
      if (suppressed.has(j) || j === i) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return indices;
}

/**
 * IoU (Intersection over Union) 計算
 */
function iou(boxA, boxB) {
  const x1 = Math.max(boxA[0], boxB[0]);
  const y1 = Math.max(boxA[1], boxB[1]);
  const x2 = Math.min(boxA[2], boxB[2]);
  const y2 = Math.min(boxA[3], boxB[3]);

  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter === 0) return 0;

  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

  return inter / (areaA + areaB - inter);
}

/**
 * 推論出力を後処理して検出結果を返す。
 */
function postprocess(output, scale, padX, padY, srcWidth, srcHeight) {
  // YOLOv8 output shape: [1, 4+num_classes, 8400] → 転置して [8400, 4+num_classes]
  const data = output.data;
  const numDetections = output.dims[2]; // 8400
  const numFeatures = output.dims[1];   // 4 + num_classes

  const detections = [];

  for (let i = 0; i < numDetections; i++) {
    // 各特徴量を取得（列優先）
    const xc = data[0 * numDetections + i];
    const yc = data[1 * numDetections + i];
    const w = data[2 * numDetections + i];
    const h = data[3 * numDetections + i];

    // クラススコア
    let maxScore = 0;
    let maxClass = 0;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = data[(4 + c) * numDetections + i];
      if (score > maxScore) {
        maxScore = score;
        maxClass = c;
      }
    }

    if (maxScore < CONF_THRESHOLD) continue;

    // 座標を元画像サイズに変換
    const x1 = (xc - w / 2 - padX) / scale;
    const y1 = (yc - h / 2 - padY) / scale;
    const x2 = (xc + w / 2 - padX) / scale;
    const y2 = (yc + h / 2 - padY) / scale;

    // クリップ
    detections.push({
      bbox: [
        Math.max(0, x1),
        Math.max(0, y1),
        Math.min(srcWidth, x2),
        Math.min(srcHeight, y2),
      ],
      classId: maxClass,
      className: CLASS_NAMES[maxClass],
      confidence: maxScore,
      points: POINT_VALUES[maxClass],
    });
  }

  // クラスごとに NMS
  const finalDetections = [];
  for (let c = 0; c < NUM_CLASSES; c++) {
    const classDetections = detections.filter(d => d.classId === c);
    if (classDetections.length === 0) continue;

    const boxes = classDetections.map(d => d.bbox);
    const scores = classDetections.map(d => d.confidence);
    const kept = nms(boxes, scores, IOU_THRESHOLD);

    for (const idx of kept) {
      finalDetections.push(classDetections[idx]);
    }
  }

  // 信頼度降順ソート
  finalDetections.sort((a, b) => b.confidence - a.confidence);

  return finalDetections;
}

/**
 * 推論を実行する。
 */
async function runInference(imageData, srcWidth, srcHeight) {
  if (!session) {
    self.postMessage({
      type: 'error',
      message: 'モデルが読み込まれていません',
    });
    return;
  }

  const startTime = performance.now();

  // 前処理
  const { tensor, scale, padX, padY } = preprocess(imageData, srcWidth, srcHeight);

  // テンソル作成
  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  // 推論実行
  const inputName = session.inputNames[0];
  const feeds = { [inputName]: inputTensor };
  const results = await session.run(feeds);

  // テンソル解放（連続モード時のメモリリーク防止）
  inputTensor.dispose();

  // 出力取得
  const outputName = session.outputNames[0];
  const output = results[outputName];

  // 後処理
  const detections = postprocess(output, scale, padX, padY, srcWidth, srcHeight);

  const elapsed = performance.now() - startTime;

  self.postMessage({
    type: 'inference-result',
    detections: detections,
    elapsed: Math.round(elapsed),
  });
}

/**
 * メッセージハンドラ
 */
self.onmessage = async function(e) {
  const msg = e.data;

  switch (msg.type) {
    case 'load-model':
      await loadModel(msg.modelData);
      break;

    case 'inference':
      await runInference(msg.imageData, msg.width, msg.height);
      break;

    default:
      self.postMessage({
        type: 'error',
        message: '不明なメッセージタイプ: ' + msg.type,
      });
  }
};
