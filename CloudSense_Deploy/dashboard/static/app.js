let cpuValues = [];
let chartInstance = null;
let autoSimInterval = null;
let fleetSyncInterval = null;

let holdTicks = 0;
let scaleOutPrompted = false;
let extraInstanceId = null; 

const ELEMENTS = {
    btnPredict: document.getElementById('btnPredictOnce'),
    toggleAuto: document.getElementById('toggleAutoSim'),
    thresholdInput: document.getElementById('thresholdInput'),
    manualCpuInput: document.getElementById('manualCpuInput'),
    
    valMAE: document.getElementById('valMAE'),
    valRMSE: document.getElementById('valRMSE'),
    valR2: document.getElementById('valR2'),
    valLookback: document.getElementById('valLookback'),
    healthStatus: document.getElementById('healthStatus'),
    modelBadge: document.getElementById('modelBadge'),
    
    currentCpuVal: document.getElementById('currentCpuVal'),
    predictedCpuVal: document.getElementById('predictedCpuVal'),
    recommendationVal: document.getElementById('recommendationVal'),
    recommendationCard: document.getElementById('recommendationCard'),
    
    fleetList: document.getElementById('fleetList'),
    fleetCountBadge: document.getElementById('fleetCountBadge'),
    
    console: document.getElementById('consoleWindow'),
    ctx: document.getElementById('cpuChart').getContext('2d'),

    modal: document.getElementById('approvalModal'),
    modalTitle: document.getElementById('modalTitle'),
    modalMsg: document.getElementById('modalMessage'),
    btnApprove: document.getElementById('btnApprove'),
    btnReject: document.getElementById('btnReject')
};

// Modals
let pendingAction = null;
function popModal(title, msg, actionCallback) {
    ELEMENTS.modalTitle.innerHTML = title;
    ELEMENTS.modalMsg.innerHTML = msg;
    pendingAction = actionCallback;
    ELEMENTS.modal.classList.add('active');
}
ELEMENTS.btnReject.onclick = () => { 
    ELEMENTS.modal.classList.remove('active'); 
    log("SYSTEM", "User rejected scaling action.", "warn");
};
ELEMENTS.btnApprove.onclick = () => {
    ELEMENTS.modal.classList.remove('active');
    if(pendingAction) pendingAction();
};


function init() {
    log("SYSTEM", "Initializing CloudSense web dashboard...", "info");
    initChart();
    fetchHealthAndMetrics();
    syncFleet();
    fleetSyncInterval = setInterval(syncFleet, 3000);
    
    for(let i=0; i<48; i++) cpuValues.push(30 + Math.random() * 5);
    updateChart(cpuValues, null);
    
    ELEMENTS.btnPredict.addEventListener('click', triggerPrediction);
    ELEMENTS.toggleAuto.addEventListener('change', (e) => {
        if(e.target.checked) {
            log("SYSTEM", "Auto-Simulator Started", "info");
            triggerPrediction();
            autoSimInterval = setInterval(triggerPrediction, 4000);
        } else {
            log("SYSTEM", "Auto-Simulator Stopped", "warn");
            clearInterval(autoSimInterval);
        }
    });
}

// AWS Mock Fleet API
async function syncFleet() {
    try {
        const r = await fetch('/api/fleet');
        const fleetDict = await r.json();
        renderFleet(fleetDict);
    } catch(e) {}
}

function renderFleet(fleetDict) {
    ELEMENTS.fleetList.innerHTML = '';
    const keys = Object.keys(fleetDict);
    ELEMENTS.fleetCountBadge.textContent = `Instances: ${keys.length}`;
    
    keys.forEach(k => {
        const inst = fleetDict[k];
        let buttons = '';
        if(inst.status === 'Running') {
            buttons = `<button class="btn btn-secondary btn-small" onclick="fleetAction('${k}','stop')">Stop</button>`;
        } else if(inst.status === 'Stopped') {
            buttons = `<button class="btn btn-secondary btn-small" onclick="fleetAction('${k}','start')">Start</button>`;
        }
        
        let killBtn = '';
        if(!inst.is_main) {
            killBtn = `<button class="btn btn-danger btn-small" onclick="fleetAction('${k}','terminate')">Kill</button>`;
        }

        const html = `
            <div class="instance-row">
                <div class="instance-info">
                    <strong><span class="inst-status ${inst.status}"></span>${inst.name}</strong>
                    <span>[${inst.id}] - ${inst.type}</span>
                </div>
                <div class="instance-actions">
                    ${buttons}
                    ${killBtn}
                </div>
            </div>
        `;
        ELEMENTS.fleetList.insertAdjacentHTML('beforeend', html);
    });
}

async function fleetAction(id, action) {
    log("FLEET", `Requesting ${action.toUpperCase()} for instance ${id}`, "req");
    if(action === 'terminate') {
        extraInstanceId = null;
        scaleOutPrompted = false;
        holdTicks = 0;
    }
    await fetch(`/api/fleet/${id}/${action}`, {method:'POST'});
    syncFleet();
}

async function provisionAWSInstance() {
    log("FLEET", "Provisioning a new Auto-Scaled Worker Instance...", "req");
    const r = await fetch('/api/fleet/provision', {method:'POST'});
    const data = await r.json();
    extraInstanceId = data.id;
    log("FLEET", `Created Instance ${data.id}`, "success");
    syncFleet();
}

// AI API Calls
async function fetchHealthAndMetrics() {
    try {
        const resHealth = await fetch('/api/health');
        const dataHealth = await resHealth.json();
        ELEMENTS.healthStatus.textContent = "● API Online";
        ELEMENTS.healthStatus.classList.remove('offline');
        ELEMENTS.modelBadge.textContent = "Model: " + dataHealth.model;
        ELEMENTS.valLookback.textContent = dataHealth.look_back;
        
        const resMet = await fetch('/api/metrics');
        const dataMet = await resMet.json();
        const m = dataMet.training_metrics;
        ELEMENTS.valMAE.textContent = parseFloat(m.mae).toFixed(3);
        ELEMENTS.valRMSE.textContent = parseFloat(m.rmse).toFixed(3);
        ELEMENTS.valR2.textContent = parseFloat(m.r2).toFixed(4);
    } catch(e) {
        ELEMENTS.healthStatus.textContent = "● Offline";
        ELEMENTS.healthStatus.classList.add('offline');
    }
}

async function triggerPrediction() {
    let nextVal;
    const manualInputStr = ELEMENTS.manualCpuInput.value;
    if (manualInputStr !== "") {
        nextVal = parseFloat(manualInputStr);
        ELEMENTS.manualCpuInput.value = ""; // clear after reading
    } else {
        const lastVal = cpuValues[cpuValues.length - 1];
        nextVal = lastVal + ((Math.random() - 0.5) * 12); 
        if (Math.random() > 0.85) nextVal += 35; // spike
        if (Math.random() > 0.85) nextVal -= 35; // drop
    }
    nextVal = Math.max(0, Math.min(100, nextVal)); 
    
    cpuValues.push(nextVal);
    if(cpuValues.length > 48) cpuValues.shift();

    const currentCpu = parseFloat(nextVal).toFixed(2);
    ELEMENTS.currentCpuVal.textContent = currentCpu + "%";
    const threshold = parseFloat(ELEMENTS.thresholdInput.value) || 70.0;
    
    try {
        const payload = { cpu_values: cpuValues, threshold: threshold };
        const res = await fetch('/api/predict/realtime', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        const predCpu = parseFloat(data.predicted_cpu_percent).toFixed(2);
        ELEMENTS.predictedCpuVal.textContent = predCpu + "%";
        ELEMENTS.recommendationVal.textContent = data.recommendation;
        
        // AUTO SCALING LOGIC
        if(data.recommendation === "SCALE_OUT") {
            ELEMENTS.recommendationCard.className = "stat-card glass-panel action-card scale_out";
            log("ALERT", `High Load Predicted -> ${predCpu}%`, "error");
            holdTicks = 0;
            
            if(!scaleOutPrompted && !extraInstanceId) {
                scaleOutPrompted = true;
                popModal(
                    "⚠️ High Load Alert", 
                    `AI Model predicted CPU reaching <b>${predCpu}%</b>.<br>Do you want to provision a new Free Tier instance to distribute the load?`, 
                    provisionAWSInstance
                );
            }
        } else {
            ELEMENTS.recommendationCard.className = "stat-card glass-panel action-card hold";
            
            if (extraInstanceId) {
                if (parseFloat(predCpu) < 40.0) {
                    holdTicks++;
                    if(holdTicks >= 3) {
                        holdTicks = 0;
                        popModal(
                            "✅ Load Normalized", 
                            `Average CPU has dropped to a safe limit (<b>${predCpu}%</b>) for consecutive ticks.<br>Do you want to terminate the extra scaled instance to save costs?`,
                            () => fleetAction(extraInstanceId, 'terminate')
                        );
                    }
                } else {
                    holdTicks = 0;
                }
            }
        }
        
        updateChart(cpuValues, parseFloat(predCpu));
    } catch (e) {
        log("ERROR", "Prediction failed", "error");
    }
}

function initChart() { /* [SAME CHART.JS CONFIG] */
    const data = {
        labels: Array.from({length: 49}, (_, i) => `T${i-48}`),
        datasets: [
            { label: 'Historical CPU Load (%)', data: [], borderColor: 'rgba(255, 255, 255, 0.4)', backgroundColor: 'rgba(255, 255, 255, 0.05)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4 },
            { label: 'Model Prediction (T+1)', data: [], borderColor: '#4facfe', backgroundColor: '#4facfe', borderWidth: 3, pointRadius: 6, pointHoverRadius: 8, pointBackgroundColor: '#00d2ff', fill: false, borderDash: [5, 5] }
        ]
    };
    chartInstance = new Chart(ELEMENTS.ctx, { type: 'line', data: data, options: { responsive: true, maintainAspectRatio: false, animation: { duration: 400 }, plugins: { legend: { labels: { color: '#8b92a5'} } }, scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b92a5', maxTicksLimit: 10 } }, y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b92a5'} } } } });
}

function updateChart(hArr, pVal) {
    if(!chartInstance) return;
    const hData = [...hArr, null];
    const pData = new Array(49).fill(null);
    if(pVal !== null) { pData[47] = hArr[hArr.length - 1]; pData[48] = pVal; }
    chartInstance.data.datasets[0].data = hData;
    chartInstance.data.datasets[1].data = pData;
    chartInstance.update();
}

function log(tag, msg, type="info") {
    const el = document.createElement('div');
    el.className = `log-line log-${type}`;
    el.innerHTML = `<span class="log-time">[${new Date().toTimeString().split(' ')[0]}]</span> <strong>[${tag}]</strong> ${msg}`;
    ELEMENTS.console.appendChild(el);
    ELEMENTS.console.scrollTop = ELEMENTS.console.scrollHeight;
}
function clearLogs() { ELEMENTS.console.innerHTML = ''; }
init();
