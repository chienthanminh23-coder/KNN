/* =====================================================================
   app.js — KNN Demo: kết nối WASM + Chart.js + UI
   ===================================================================== */

// ---- Biến toàn cục -------------------------------------------------
let knn       = null;    // KNNClassifier instance (từ WASM)
let chart     = null;    // Chart.js scatter plot
let trainData = [];      // Dữ liệu train
let testData  = [];      // Dữ liệu test

const CLASS_NAMES  = IRIS_DATA.classNames;
const CLASS_COLORS = IRIS_DATA.classColors;
const FEAT_NAMES   = IRIS_DATA.featureNames;

// ---- Khởi động WASM ------------------------------------------------
async function initWasm() {
  // Nạp module WASM (knn.js do Emscripten sinh ra)
  const Module = await KNNModule();
  knn = new Module.KNNClassifier();

  // Chia dữ liệu train/test
  const split = splitTrainTest(IRIS_DATA.samples, 0.3, 42);
  trainData = split.train;
  testData  = split.test;

  // Chuẩn bị dữ liệu cho fit()
  const flatX  = trainData.flatMap(s => s.slice(0, 4));
  const labels = trainData.map(s => s[4]);
  const n = trainData.length, f = 4;

  knn.fit(flatX, n, f, labels);

  // Ẩn loading, hiện app
  document.getElementById('loading').style.display = 'none';
  document.getElementById('app').style.display     = 'block';

  // Khởi tạo UI
  updateStats();
  initChart();
  setupEventListeners();
}

// ---- Cập nhật thống kê ---------------------------------------------
function updateStats() {
  const k = parseInt(document.getElementById('k-slider').value);

  // Tính accuracy trên test set
  const flatX  = testData.flatMap(s => s.slice(0, 4));
  const labels = testData.map(s => s[4]);
  const acc    = knn.score(flatX, testData.length, 4, labels, k);

  document.getElementById('stat-train').textContent = trainData.length;
  document.getElementById('stat-test').textContent  = testData.length;
  document.getElementById('stat-k').textContent     = k;
  document.getElementById('stat-acc').textContent   = acc.toFixed(1) + '%';
}

// ---- Khởi tạo scatter plot -----------------------------------------
function initChart() {
  const ctx = document.getElementById('scatter-chart').getContext('2d');

  const xFeat = parseInt(document.getElementById('x-axis').value);
  const yFeat = parseInt(document.getElementById('y-axis').value);

  // Tạo datasets theo lớp
  const datasets = CLASS_NAMES.map((name, ci) => ({
    label: name,
    data: trainData
      .filter(s => s[4] === ci)
      .map(s => ({ x: s[xFeat], y: s[yFeat] })),
    backgroundColor: CLASS_COLORS[ci] + 'cc',
    borderColor: CLASS_COLORS[ci],
    borderWidth: 1.5,
    pointRadius: 5,
    pointHoverRadius: 7,
  }));

  // Dataset cho điểm truy vấn (ban đầu rỗng)
  datasets.push({
    label: 'Query Point',
    data: [],
    backgroundColor: '#fff',
    borderColor: '#FFD700',
    borderWidth: 3,
    pointRadius: 10,
    pointStyle: 'star',
    pointHoverRadius: 14,
  });

  chart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: {
          labels: { color: '#e8eaf6', font: { size: 12 } }
        },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              return `${ctx.dataset.label}: (${d.x.toFixed(2)}, ${d.y.toFixed(2)})`;
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: FEAT_NAMES[xFeat] + ' (cm)', color: '#7986cb' },
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#7986cb' }
        },
        y: {
          title: { display: true, text: FEAT_NAMES[yFeat] + ' (cm)', color: '#7986cb' },
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#7986cb' }
        }
      }
    }
  });
}

// ---- Cập nhật chart khi đổi trục ----------------------------------
function updateChart() {
  if (!chart) return;
  const xFeat = parseInt(document.getElementById('x-axis').value);
  const yFeat = parseInt(document.getElementById('y-axis').value);

  CLASS_NAMES.forEach((_, ci) => {
    chart.data.datasets[ci].data = trainData
      .filter(s => s[4] === ci)
      .map(s => ({ x: s[xFeat], y: s[yFeat] }));
  });
  chart.options.scales.x.title.text = FEAT_NAMES[xFeat] + ' (cm)';
  chart.options.scales.y.title.text = FEAT_NAMES[yFeat] + ' (cm)';
  chart.update();
}

// ---- Dự đoán -------------------------------------------------------
function predict() {
  if (!knn) return;

  const features = [
    parseFloat(document.getElementById('in-sl').value),
    parseFloat(document.getElementById('in-sw').value),
    parseFloat(document.getElementById('in-pl').value),
    parseFloat(document.getElementById('in-pw').value),
  ];

  if (features.some(isNaN)) {
    alert('Vui lòng nhập đầy đủ 4 giá trị features!');
    return;
  }

  const k = parseInt(document.getElementById('k-slider').value);

  // Gọi C++ qua WASM
  const label     = knn.predict(features, k);
  const probaJSON = knn.predictProba(features, k);
  const nbJSON    = knn.kNeighbors(features, k);
  const proba     = JSON.parse(probaJSON);
  const neighbors = JSON.parse(nbJSON);

  // Hiện kết quả
  showResult(label, proba);
  showNeighbors(neighbors);
  updateQueryPoint(features);
}

// ---- Hiển thị kết quả dự đoán -------------------------------------
function showResult(label, proba) {
  const box = document.getElementById('result-box');
  box.classList.add('show');
  box.querySelector('.result-class').textContent = CLASS_NAMES[label];
  box.querySelector('.result-class').style.color = CLASS_COLORS[label];

  const probaDiv = box.querySelector('.result-proba');
  probaDiv.innerHTML = '';
  Object.entries(proba).forEach(([cls, p]) => {
    const chip = document.createElement('span');
    chip.className = 'proba-chip';
    chip.style.background = CLASS_COLORS[parseInt(cls)] + '44';
    chip.style.border = `1px solid ${CLASS_COLORS[parseInt(cls)]}`;
    chip.style.color  = CLASS_COLORS[parseInt(cls)];
    chip.textContent  = `${CLASS_NAMES[parseInt(cls)]}: ${(p * 100).toFixed(0)}%`;
    probaDiv.appendChild(chip);
  });
}

// ---- Hiển thị bảng k láng giềng -----------------------------------
function showNeighbors(neighbors) {
  const tbody = document.querySelector('#neighbors-table tbody');
  tbody.innerHTML = '';
  neighbors.forEach((nb, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${nb.distance.toFixed(4)}</td>
      <td><span class="class-badge" style="background:${CLASS_COLORS[nb.label]}88;border:1px solid ${CLASS_COLORS[nb.label]}">
        ${CLASS_NAMES[nb.label]}
      </span></td>`;
    tbody.appendChild(tr);
  });
}

// ---- Vẽ điểm truy vấn lên chart ------------------------------------
function updateQueryPoint(features) {
  if (!chart) return;
  const xFeat = parseInt(document.getElementById('x-axis').value);
  const yFeat = parseInt(document.getElementById('y-axis').value);
  // Dataset cuối cùng là query point
  const qIdx = chart.data.datasets.length - 1;
  chart.data.datasets[qIdx].data = [{ x: features[xFeat], y: features[yFeat] }];
  chart.update();
}

// ---- Gắn event listeners -------------------------------------------
function setupEventListeners() {
  // Slider k
  const slider  = document.getElementById('k-slider');
  const kDisplay = document.getElementById('k-display');
  slider.addEventListener('input', () => {
    kDisplay.textContent = slider.value;
    updateStats();
  });

  // Đổi trục
  document.getElementById('x-axis').addEventListener('change', updateChart);
  document.getElementById('y-axis').addEventListener('change', updateChart);

  // Nút dự đoán
  document.getElementById('btn-predict').addEventListener('click', predict);

  // Enter để dự đoán
  ['in-sl','in-sw','in-pl','in-pw'].forEach(id => {
    document.getElementById(id).addEventListener('keydown', e => {
      if (e.key === 'Enter') predict();
    });
  });
}

// ---- Start ---------------------------------------------------------
window.addEventListener('load', initWasm);
