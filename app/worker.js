/**
 * シールカウンター - Web Worker
 * ONNX Runtime Web による YOLOv8-nano 推論 + 年号検出CNN実行
 */

// ONNX Runtime Web CDN
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js');

// 定数 — YOLO
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.15;
const IOU_THRESHOLD = 0.65;
const NUM_CLASSES = 6;
const PADDING_COLOR = 114; // RGB(114,114,114) — Ultralytics デフォルト

const CLASS_NAMES = [
  'seal_05', 'seal_1', 'seal_15', 'seal_2',
  'seal_25', 'seal_3'
];

const POINT_VALUES = [0.5, 1, 1.5, 2, 2.5, 3];

// 定数 — 年号検出CNN
const YEAR_CNN_INPUT_W = 96;
const YEAR_CNN_INPUT_H = 48;
const YEAR_REGION_RATIO = 0.35;   // クロップ上部35%が年号領域
const YEAR_CONF_THRESHOLD = 0.85; // この信頼度未満は "unknown"
// *** 毎年更新: 新年度のシールPNG取得→CNNモデル再訓練→ONNX再エクスポート後に更新 ***
const CURRENT_YEAR = '2026';
const YEAR_CLASSES = ['2023', '2024', '2025', '2026'];

let session = null;
let yearSession = null;

/**
 * YOLOモデルを読み込む。
 */
async function loadModel(modelData) {
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

    const options = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    };

    session = await ort.InferenceSession.create(modelData, options);

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
 * 年号検出CNNモデルを読み込む。
 */
async function loadYearModel(modelData) {
  try {
    const options = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    };

    yearSession = await ort.InferenceSession.create(modelData, options);

    self.postMessage({
      type: 'year-model-loaded',
    });
  } catch (e) {
    // 年号モデル読み込み失敗は致命的ではない（旧年度判定なしで動作）
    self.postMessage({
      type: 'year-model-error',
      message: '年号モデル読み込みに失敗: ' + e.message,
    });
  }
}

/**
 * RGBA画像データから指定領域をグレースケール96x48に変換し、年号CNNで推論する。
 * Returns: { year: string, confidence: number } or null
 */
async function classifyYear(imageData, srcWidth, bbox) {
  if (!yearSession) return null;

  const [x1, y1, x2, y2] = bbox;
  const cropW = Math.round(x2 - x1);
  const cropH = Math.round(y2 - y1);
  if (cropW < 8 || cropH < 8) return null;

  // 年号領域: クロップ上部35%
  const yearH = Math.round(cropH * YEAR_REGION_RATIO);
  const yearX1 = Math.round(x1);
  const yearY1 = Math.round(y1);
  const yearX2 = Math.round(x2);
  const yearY2 = Math.round(y1 + yearH);

  // 年号領域をグレースケール96x48に変換（bilinear補間）
  const regionW = yearX2 - yearX1;
  const regionH = yearY2 - yearY1;
  if (regionW < 4 || regionH < 4) return null;

  const pixels = imageData.data; // RGBA
  const input = new Float32Array(YEAR_CNN_INPUT_H * YEAR_CNN_INPUT_W);

  for (let dy = 0; dy < YEAR_CNN_INPUT_H; dy++) {
    // Bilinear補間: 連続座標を計算
    const srcYf = yearY1 + (dy + 0.5) * regionH / YEAR_CNN_INPUT_H - 0.5;
    const sy0 = Math.max(0, Math.floor(srcYf));
    const sy1 = Math.min(sy0 + 1, yearY2 - 1);
    const fy = srcYf - sy0;

    for (let dx = 0; dx < YEAR_CNN_INPUT_W; dx++) {
      const srcXf = yearX1 + (dx + 0.5) * regionW / YEAR_CNN_INPUT_W - 0.5;
      const sx0 = Math.max(0, Math.floor(srcXf));
      const sx1 = Math.min(sx0 + 1, yearX2 - 1);
      const fx = srcXf - sx0;

      // 4隣接ピクセルのグレースケール値
      const idx00 = (sy0 * srcWidth + sx0) * 4;
      const idx10 = (sy0 * srcWidth + sx1) * 4;
      const idx01 = (sy1 * srcWidth + sx0) * 4;
      const idx11 = (sy1 * srcWidth + sx1) * 4;

      const g00 = 0.299 * pixels[idx00] + 0.587 * pixels[idx00 + 1] + 0.114 * pixels[idx00 + 2];
      const g10 = 0.299 * pixels[idx10] + 0.587 * pixels[idx10 + 1] + 0.114 * pixels[idx10 + 2];
      const g01 = 0.299 * pixels[idx01] + 0.587 * pixels[idx01 + 1] + 0.114 * pixels[idx01 + 2];
      const g11 = 0.299 * pixels[idx11] + 0.587 * pixels[idx11 + 1] + 0.114 * pixels[idx11 + 2];

      // Bilinear補間 + 正規化 [0,1]
      const gray = ((1 - fx) * (1 - fy) * g00 + fx * (1 - fy) * g10
                   + (1 - fx) * fy * g01 + fx * fy * g11) / 255.0;
      input[dy * YEAR_CNN_INPUT_W + dx] = gray;
    }
  }

  // 推論: input shape [1, 1, 48, 96]
  const tensor = new ort.Tensor('float32', input, [1, 1, YEAR_CNN_INPUT_H, YEAR_CNN_INPUT_W]);
  const inputName = yearSession.inputNames[0];
  const results = await yearSession.run({ [inputName]: tensor });
  tensor.dispose();

  const outputName = yearSession.outputNames[0];
  const yearOutput = results[outputName];
  const logits = yearOutput.data; // Float32Array [4]
  // Note: logits is a view into yearOutput's buffer, so copy before dispose
  const logitsCopy = new Float32Array(logits);
  yearOutput.dispose();

  // Softmax
  let maxLogit = -Infinity;
  for (let i = 0; i < logitsCopy.length; i++) {
    if (logitsCopy[i] > maxLogit) maxLogit = logitsCopy[i];
  }
  let sumExp = 0;
  const probs = new Float32Array(logitsCopy.length);
  for (let i = 0; i < logitsCopy.length; i++) {
    probs[i] = Math.exp(logitsCopy[i] - maxLogit);
    sumExp += probs[i];
  }
  let bestIdx = 0;
  let bestProb = 0;
  for (let i = 0; i < probs.length; i++) {
    probs[i] /= sumExp;
    if (probs[i] > bestProb) {
      bestProb = probs[i];
      bestIdx = i;
    }
  }

  return {
    year: YEAR_CLASSES[bestIdx],
    confidence: bestProb,
  };
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

  // Agnostic NMS（クラス無関係に重複排除）
  // 同一領域に異なるクラスの検出が出る問題を解消
  const boxes = detections.map(d => d.bbox);
  const scores = detections.map(d => d.confidence);
  const kept = nms(boxes, scores, IOU_THRESHOLD);

  const finalDetections = kept.map(idx => detections[idx]);

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
  const preprocessed = preprocess(imageData, srcWidth, srcHeight);

  // テンソル作成（前処理配列の所有権をTensorに委譲）
  const inputTensor = new ort.Tensor('float32', preprocessed.tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);

  // 推論実行
  const inputName = session.inputNames[0];
  const feeds = { [inputName]: inputTensor };
  const results = await session.run(feeds);

  // テンソル解放
  inputTensor.dispose();

  // 出力取得 → 後処理 → 出力テンソル解放
  const outputName = session.outputNames[0];
  const output = results[outputName];
  const detections = postprocess(output, preprocessed.scale, preprocessed.padX, preprocessed.padY, srcWidth, srcHeight);
  output.dispose();

  // 年号判定: 各検出に対してCNNで年号を分類
  if (yearSession) {
    for (const det of detections) {
      const yearResult = await classifyYear(imageData, srcWidth, det.bbox);
      if (yearResult && yearResult.confidence >= YEAR_CONF_THRESHOLD
          && yearResult.year !== CURRENT_YEAR) {
        // 旧年度シール: classId=6 にマーク
        det.classId = 6;
        det.className = 'seal_old';
        det.yearDetected = yearResult.year;
        det.yearConfidence = yearResult.confidence;
      } else if (yearResult) {
        det.yearDetected = yearResult.year;
        det.yearConfidence = yearResult.confidence;
      }
    }
  }

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

    case 'load-year-model':
      await loadYearModel(msg.modelData);
      break;

    case 'inference':
      const imgData = { data: new Uint8ClampedArray(msg.imageBuffer) };
      await runInference(imgData, msg.width, msg.height);
      break;

    default:
      self.postMessage({
        type: 'error',
        message: '不明なメッセージタイプ: ' + msg.type,
      });
  }
};
