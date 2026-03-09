#!/usr/bin/env python3
"""
GNINA Molecular Docking Web Application
Optimized for 36 CPUs + 2x RTX 5000 GPUs

Features:
- Parallel ligand preparation with proper protonation
- Dual GPU load balancing
- Real-time progress tracking via WebSocket
- Automatic binding site detection from reference ligand
- Support for SMILES input and SDF file batches
"""

import asyncio
import os
import tempfile
import subprocess
import shutil
import json
import logging
import logging.handlers
import uuid
import io
import zipfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdShapeHelpers, DataStructs
from rdkit import RDLogger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging — console + rotating file
_LOG_FILE = os.path.join(os.path.dirname(__file__), 'gnina_app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            _LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8'
        ),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# EMBEDDED HTML (fallback if templates/index.html not found)
# ============================================================================
EMBEDDED_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNINA Molecular Docking</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
        }
        .container { max-width: 1000px; margin: 0 auto; padding: 2rem; }
        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--bg-card) 0%, #0f172a 100%);
            border-radius: 16px;
            border: 1px solid var(--border);
        }
        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .header p { color: var(--text-muted); font-size: 0.95rem; }
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        .stat {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid rgba(37, 99, 235, 0.3);
            border-radius: 8px;
            font-size: 0.85rem;
        }
        .form-card {
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border);
            padding: 2rem;
            margin-bottom: 1.5rem;
        }
        .form-card h2 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        @media (max-width: 768px) { .form-grid { grid-template-columns: 1fr; } }
        .form-group { margin-bottom: 1rem; }
        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .required::after { content: '*'; color: var(--error); margin-left: 4px; }
        input[type="file"] {
            width: 100%;
            padding: 0.75rem;
            background: var(--bg-dark);
            border: 2px dashed var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        input[type="file"]:hover, input[type="file"]:focus {
            border-color: var(--primary);
            outline: none;
        }
        input[type="file"]::file-selector-button {
            padding: 0.5rem 1rem;
            margin-right: 1rem;
            background: var(--primary);
            border: none;
            border-radius: 6px;
            color: white;
            font-weight: 500;
            cursor: pointer;
        }
        input[type="number"], select {
            width: 100%;
            padding: 0.75rem;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        input[type="number"]:focus, select:focus {
            border-color: var(--primary);
            outline: none;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
        }
        textarea {
            width: 100%;
            min-height: 200px;
            padding: 1rem;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.85rem;
            resize: vertical;
            transition: all 0.2s;
        }
        textarea:focus {
            border-color: var(--primary);
            outline: none;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
        }
        textarea::placeholder { color: var(--text-muted); opacity: 0.6; }
        .input-toggle {
            display: flex;
            background: var(--bg-dark);
            border-radius: 8px;
            padding: 4px;
            margin-bottom: 1rem;
        }
        .toggle-btn {
            flex: 1;
            padding: 0.6rem 1rem;
            background: transparent;
            border: none;
            border-radius: 6px;
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .toggle-btn.active { background: var(--primary); color: white; }
        .toggle-btn:hover:not(.active) { color: var(--text); }
        .options-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }
        .gpu-status { display: flex; gap: 1rem; margin-top: 1rem; }
        .gpu-card {
            flex: 1;
            padding: 1rem;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            text-align: center;
        }
        .gpu-card.active { border-color: var(--success); }
        .gpu-name { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem; }
        .gpu-card.active .gpu-name { color: var(--success); }
        .gpu-info { font-size: 0.75rem; color: var(--text-muted); }
        .submit-btn {
            width: 100%;
            padding: 1rem 2rem;
            background: linear-gradient(135deg, var(--primary) 0%, #7c3aed 100%);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
        }
        .submit-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .progress-panel {
            display: none;
            background: var(--bg-card);
            border-radius: 16px;
            border: 1px solid var(--border);
            padding: 2rem;
            margin-top: 1.5rem;
        }
        .progress-panel.visible { display: block; }
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        .progress-title { font-size: 1.2rem; font-weight: 600; }
        .progress-status {
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }
        .progress-status.preparing { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
        .progress-status.docking { background: rgba(37, 99, 235, 0.2); color: var(--primary); }
        .progress-status.completed { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .progress-status.failed { background: rgba(239, 68, 68, 0.2); color: var(--error); }
        .progress-bar-container {
            background: var(--bg-dark);
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--success));
            width: 0%;
            transition: width 0.5s ease;
            border-radius: 8px;
        }
        .progress-message { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
        .progress-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }
        .progress-detail { padding: 0.75rem; background: var(--bg-dark); border-radius: 8px; }
        .progress-detail-label { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.25rem; }
        .progress-detail-value { font-size: 1rem; font-weight: 600; font-family: monospace; }
        .alert {
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            display: none;
        }
        .alert.visible { display: flex; align-items: flex-start; gap: 0.75rem; }
        .alert-error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
        }
        .alert-success {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #6ee7b7;
        }
        .help-text { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem; }
        .file-info { margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-muted); }
        .file-info.valid { color: var(--success); }
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 640px) {
            .container { padding: 1rem; }
            .header { padding: 1.5rem; }
            .header h1 { font-size: 1.5rem; }
            .form-card { padding: 1.25rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🧬 GNINA Molecular Docking</h1>
            <p>GPU-accelerated molecular docking with deep learning scoring</p>
            <div class="stats-bar">
                <div class="stat"><span>⚡</span><span>36 CPU Cores</span></div>
                <div class="stat"><span>🎮</span><span>2× RTX 5000 GPUs</span></div>
                <div class="stat"><span>🔬</span><span>CNN Scoring</span></div>
            </div>
        </header>
        <div id="alertError" class="alert alert-error">
            <span>⚠️</span><span id="alertErrorMessage"></span>
        </div>
        <div id="alertSuccess" class="alert alert-success">
            <span>✅</span><span id="alertSuccessMessage"></span>
        </div>
        <form id="dockingForm">
            <div class="form-card">
                <h2>📁 Input Structures</h2>
                <div class="form-grid">
                    <div class="form-group">
                        <label class="required">Protein Target (PDB)</label>
                        <input type="file" name="receptor" id="receptorFile" accept=".pdb" required>
                        <div id="receptorInfo" class="file-info"></div>
                    </div>
                    <div class="form-group">
                        <label class="required">Reference Ligand (SDF)</label>
                        <input type="file" name="reference" id="referenceFile" accept=".sdf,.mol" required>
                        <div class="help-text">Defines the binding site location (autobox)</div>
                    </div>
                </div>
            </div>
            <div class="form-card">
                <h2>⚗️ Ligands to Dock</h2>
                <div class="input-toggle">
                    <button type="button" class="toggle-btn active" data-input="smiles">SMILES Input</button>
                    <button type="button" class="toggle-btn" data-input="file">SDF File</button>
                </div>
                <div id="smilesInputSection">
                    <div class="form-group">
                        <label>Enter SMILES (one per line)</label>
                        <textarea name="ligand_smiles" id="smilesInput" placeholder="CCO ethanol&#10;c1ccccc1,benzene&#10;CC(=O)Oc1ccccc1C(=O)O aspirin&#10;CC(C)Cc1ccc(cc1)C(C)C(=O)O,ibuprofen"></textarea>
                        <div class="help-text">
                            <span id="smilesCount">0</span> SMILES detected • Supports: SMILES, SMILES ID, or SMILES,ID format
                        </div>
                    </div>
                </div>
                <div id="fileInputSection" style="display: none;">
                    <div class="form-group">
                        <label>Upload SDF File</label>
                        <input type="file" name="ligand_file" id="ligandFile" accept=".sdf">
                        <div id="ligandInfo" class="file-info"></div>
                    </div>
                </div>
            </div>
            <div class="form-card">
                <h2>⚙️ Docking Parameters</h2>
                <div class="options-grid">
                    <div class="form-group">
                        <label>pH for Protonation</label>
                        <input type="number" name="ph" value="7.4" step="0.1" min="0" max="14">
                    </div>
                    <div class="form-group">
                        <label>Poses per Ligand</label>
                        <input type="number" name="num_poses" value="9" min="1" max="50">
                    </div>
                    <div class="form-group">
                        <label>Exhaustiveness</label>
                        <input type="number" name="exhaustiveness" value="8" min="1" max="64">
                    </div>
                    <div class="form-group">
                        <label>CNN Scoring</label>
                        <select name="cnn_scoring">
                            <option value="rescore" selected>Rescore (Fast)</option>
                            <option value="refine">Refine (Accurate)</option>
                            <option value="score_only">Score Only</option>
                            <option value="none">None (AutoDock only)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Sort Results By</label>
                        <select name="sort_by">
                            <option value="minimizedAffinity" selected>Vina Affinity (lower=better)</option>
                            <option value="CNNaffinity">CNN Affinity (higher=better)</option>
                            <option value="CNN_VS">CNN_VS Score (higher=better)</option>
                        </select>
                    </div>
                </div>
                <div class="gpu-status">
                    <div class="gpu-card active">
                        <div class="gpu-name">GPU 0 • RTX 5000</div>
                        <div class="gpu-info">16 CPU cores assigned</div>
                    </div>
                    <div class="gpu-card active">
                        <div class="gpu-name">GPU 1 • RTX 5000</div>
                        <div class="gpu-info">16 CPU cores assigned</div>
                    </div>
                </div>
            </div>
            <button type="submit" class="submit-btn" id="submitBtn">
                <span id="submitText">🚀 Start Docking</span>
                <div class="spinner" id="submitSpinner" style="display: none;"></div>
            </button>
        </form>
        <div class="progress-panel" id="progressPanel">
            <div class="progress-header">
                <span class="progress-title">Docking Progress</span>
                <span class="progress-status" id="progressStatus">Preparing</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="progress-message" id="progressMessage">Initializing...</div>
            <div class="progress-details">
                <div class="progress-detail">
                    <div class="progress-detail-label">Stage</div>
                    <div class="progress-detail-value" id="currentStage">—</div>
                </div>
                <div class="progress-detail">
                    <div class="progress-detail-label">Ligands</div>
                    <div class="progress-detail-value" id="ligandProgress">0 / 0</div>
                </div>
                <div class="progress-detail">
                    <div class="progress-detail-label">Preparation</div>
                    <div class="progress-detail-value" id="timePrepare">—</div>
                </div>
                <div class="progress-detail">
                    <div class="progress-detail-label">Docking</div>
                    <div class="progress-detail-value" id="timeDocking">—</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        let currentInputMode = 'smiles';
        const form = document.getElementById('dockingForm');
        const submitBtn = document.getElementById('submitBtn');
        const submitText = document.getElementById('submitText');
        const submitSpinner = document.getElementById('submitSpinner');
        const progressPanel = document.getElementById('progressPanel');
        const smilesInput = document.getElementById('smilesInput');
        const alertError = document.getElementById('alertError');
        const alertSuccess = document.getElementById('alertSuccess');

        document.querySelectorAll('.toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentInputMode = btn.dataset.input;
                document.getElementById('smilesInputSection').style.display = currentInputMode === 'smiles' ? 'block' : 'none';
                document.getElementById('fileInputSection').style.display = currentInputMode === 'file' ? 'block' : 'none';
            });
        });

        smilesInput.addEventListener('input', () => {
            const lines = smilesInput.value.split('\\n').filter(line => line.trim());
            document.getElementById('smilesCount').textContent = lines.length;
        });

        document.getElementById('receptorFile').addEventListener('change', (e) => {
            const file = e.target.files[0];
            const info = document.getElementById('receptorInfo');
            if (file) {
                info.textContent = '✓ ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
                info.classList.add('valid');
            } else {
                info.textContent = '';
                info.classList.remove('valid');
            }
        });

        document.getElementById('ligandFile').addEventListener('change', (e) => {
            const file = e.target.files[0];
            const info = document.getElementById('ligandInfo');
            if (file) {
                info.textContent = '✓ ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
                info.classList.add('valid');
            } else {
                info.textContent = '';
                info.classList.remove('valid');
            }
        });

        function showAlert(type, message) {
            const alert = type === 'error' ? alertError : alertSuccess;
            const msgEl = type === 'error' ? document.getElementById('alertErrorMessage') : document.getElementById('alertSuccessMessage');
            alertError.classList.remove('visible');
            alertSuccess.classList.remove('visible');
            msgEl.textContent = message;
            alert.classList.add('visible');
            if (type === 'success') setTimeout(() => alert.classList.remove('visible'), 5000);
        }

        function hideAlerts() {
            alertError.classList.remove('visible');
            alertSuccess.classList.remove('visible');
        }

        function updateProgress(data) {
            document.getElementById('progressBar').style.width = data.progress + '%';
            document.getElementById('progressMessage').textContent = data.message || 'Processing...';
            const statusEl = document.getElementById('progressStatus');
            statusEl.textContent = data.status;
            statusEl.className = 'progress-status ' + data.status;
            if (data.current_stage) document.getElementById('currentStage').textContent = data.current_stage;
            if (data.total_ligands > 0) {
                document.getElementById('ligandProgress').textContent = (data.processed_ligands || 0) + ' / ' + data.total_ligands;
            }
            if (data.timings) {
                if (data.timings.preparation) document.getElementById('timePrepare').textContent = data.timings.preparation.toFixed(1) + 's';
                if (data.timings.docking) document.getElementById('timeDocking').textContent = data.timings.docking.toFixed(1) + 's';
            }
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideAlerts();
            const receptorFile = document.getElementById('receptorFile').files[0];
            const referenceFile = document.getElementById('referenceFile').files[0];
            if (!receptorFile) { showAlert('error', 'Please upload a receptor PDB file'); return; }
            if (!referenceFile) { showAlert('error', 'Please upload a reference ligand SDF file'); return; }
            if (currentInputMode === 'smiles') {
                if (!smilesInput.value.trim()) { showAlert('error', 'Please enter at least one SMILES string'); return; }
            } else {
                const ligandFile = document.getElementById('ligandFile').files[0];
                if (!ligandFile) { showAlert('error', 'Please upload a ligand SDF file'); return; }
            }
            const formData = new FormData(form);
            if (currentInputMode === 'smiles') formData.delete('ligand_file');
            else formData.delete('ligand_smiles');
            submitBtn.disabled = true;
            submitText.textContent = 'Processing...';
            submitSpinner.style.display = 'block';
            progressPanel.classList.add('visible');
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('progressMessage').textContent = 'Submitting job...';
            document.getElementById('progressStatus').textContent = 'pending';
            document.getElementById('progressStatus').className = 'progress-status';
            document.getElementById('currentStage').textContent = '—';
            document.getElementById('ligandProgress').textContent = '0 / 0';
            document.getElementById('timePrepare').textContent = '—';
            document.getElementById('timeDocking').textContent = '—';
            try {
                const response = await fetch('/dock', { method: 'POST', body: formData });
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    const disposition = response.headers.get('content-disposition');
                    let filename = 'docking_results.sdf';
                    if (disposition) {
                        const match = disposition.match(/filename="?([^"]+)"?/);
                        if (match) filename = match[1];
                    }
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    updateProgress({ status: 'completed', progress: 100, message: 'Docking complete! Results downloaded.' });
                    showAlert('success', 'Docking completed successfully! Check your downloads.');
                } else {
                    let errorMsg = 'Docking failed';
                    try {
                        const errorData = await response.json();
                        errorMsg = errorData.detail || errorData.error || errorMsg;
                    } catch { errorMsg = await response.text() || errorMsg; }
                    updateProgress({ status: 'failed', progress: 0, message: errorMsg });
                    showAlert('error', errorMsg);
                }
            } catch (error) {
                updateProgress({ status: 'failed', progress: 0, message: error.message });
                showAlert('error', 'Request failed: ' + error.message);
            } finally {
                submitBtn.disabled = false;
                submitText.textContent = '🚀 Start Docking';
                submitSpinner.style.display = 'none';
            }
        });
    </script>
</body>
</html>'''

# ============================================================================
# CONFIGURATION - Optimized for 36 CPUs + 2x RTX 5000
# ============================================================================
N_CPU = 36
N_GPU = 1                          # Number of GPUs to use for docking
DOCK_GPU_ID = int(os.environ.get('DOCK_GPU_ID', '1'))  # Physical GPU index (GPU 1 = free RTX 5000)
RESERVED_CPU = 4                   # Reserve for system/web server
WORKER_CPU = N_CPU - RESERVED_CPU  # 32 available workers
CPU_PER_GPU = WORKER_CPU // N_GPU  # 32 cores for the single docking GPU

# Working directories
WORK_DIR = Path("/tmp/gnina_work")
WORK_DIR.mkdir(exist_ok=True)

# GNINA binary path - update as needed
GNINA_PATH = os.environ.get('GNINA_PATH', '/opt/gnina/gnina.1.3.2')

# PyMOL binary for headless session generation (pymol -cq script.py)
PYMOL_PATH = os.environ.get('PYMOL_PATH', '/home/evehom/Programs/miniconda3/envs/pymol/bin/pymol')

# ============================================================================
# DATA CLASSES
# ============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing_ligands"
    DOCKING = "docking"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class JobProgress:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    total_ligands: int = 0
    processed_ligands: int = 0
    current_stage: str = ""
    error: Optional[str] = None
    result_file: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)

# Global job tracking
active_jobs: Dict[str, JobProgress] = {}
job_websockets: Dict[str, List[WebSocket]] = {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def secure_filename(filename: str) -> str:
    """Secure a filename by removing dangerous characters."""
    if not filename:
        return "unnamed"
    filename = unicodedata.normalize('NFKD', filename)
    filename = re.sub(r'[^\w\s\-.]', '_', filename)
    filename = re.sub(r'[\s]+', '_', filename)
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def sanitize_pymol_name(name: str) -> str:
    """Sanitize a string for use as a PyMOL object name."""
    name = re.sub(r'[^\w\-]', '_', str(name))
    if name and name[0].isdigit():
        name = 'mol_' + name
    return name[:50] or 'mol'


# Priority-ordered list of SDF data field names to check for a molecule identifier.
# Spaces in field names are supported; the extracted value has spaces replaced with underscores.
_NAME_FIELDS = (
    'Structure ID', 'Structure_ID',
    'Molecule Name', 'Molecule_Name',
    'Name', 'MOLNAME', 'molname',
    'Title', 'ID', 'Compound', 'CAS',
)


def _extract_mol_name(mol, fallback: str) -> str:
    """
    Extract the best available molecule name from an RDKit mol object.

    Checks (in order):
      1. The SDF title line (_Name property)
      2. Common SDF data fields (see _NAME_FIELDS)
      3. fallback string (e.g. 'mol_42')

    Spaces in the returned name are replaced with underscores so the name is
    safe for use in SDF title lines, file names, and PyMOL object names.
    """
    name = mol.GetProp('_Name').strip() if mol.HasProp('_Name') else ''
    if not name:
        for field in _NAME_FIELDS:
            if mol.HasProp(field):
                name = mol.GetProp(field).strip()
                if name:
                    break
    if not name:
        return fallback
    # Replace spaces (and other whitespace) with underscores
    return re.sub(r'\s+', '_', name)


def _has_3d_coords(mol) -> bool:
    """Return True if the RDKit molecule has non-zero Z coordinates (i.e. is 3D)."""
    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    return any(abs(conf.GetAtomPosition(i).z) > 0.001
               for i in range(mol.GetNumAtoms()))


def _fix_split_sdf_blocks(content: str) -> str:
    """
    GNINA writes docking scores in a separate $$$$-delimited block after the structure:

        [mol name]              <- structure block: has 3D connection table, no score props
        [connection table]
        M  END
        $$$$
        [mol name]              <- properties block: has 0-atom connection table + score props
        [0-atom connection table]
        M  END
        > <minimizedAffinity>
        -5.95
        ...
        $$$$

    The properties block repeats the molecule name on the first line (does NOT start
    with '>').  We detect such split pairs by score-property presence and merge them,
    keeping only the data fields from the properties block (skipping its connection table).
    """
    # Match score properties regardless of whitespace between '>' and '<'
    # GNINA sometimes writes '>  <minimizedAffinity>' (two spaces)
    _SCORE_MARKERS = ('minimizedAffinity>', 'CNNscore>', 'CNNaffinity>')

    def _props_only(block: str) -> str:
        """Return only the '> <field>' sections from a block, stripping the connection table.
        Handles GNINA's '>  <field>' format (two spaces between > and <)."""
        m = re.search(r'>\s*<', block)
        if not m:
            return ''
        return block[m.start():].strip()

    raw = [b for b in content.split('$$$$') if b.strip()]
    merged = []
    i = 0
    while i < len(raw):
        current = raw[i].strip()
        # Merge as long as the next block carries scores that the current block lacks
        while i + 1 < len(raw):
            next_b = raw[i + 1]
            next_has_scores = any(m in next_b for m in _SCORE_MARKERS)
            current_has_scores = any(m in current for m in _SCORE_MARKERS)
            if next_has_scores and not current_has_scores:
                i += 1
                current = current + '\n' + _props_only(raw[i])
            else:
                break
        merged.append(current)
        i += 1
    return ('\n$$$$\n'.join(merged) + '\n$$$$\n') if merged else ''


def parse_smiles_input(smiles_text: str) -> List[Tuple[str, str]]:
    """
    Parse SMILES input, supporting optional identifiers.
    
    Formats supported:
    - SMILES only: CCO
    - SMILES with space-separated ID: CCO ethanol
    - SMILES with comma-separated ID: CCO,ethanol
    - SMILES with tab-separated ID: CCO\tethanol
    
    Returns: List of (smiles, identifier) tuples
    """
    results = []
    
    # Normalize line endings (handle Windows \r\n and old Mac \r)
    smiles_text = smiles_text.replace('\r\n', '\n').replace('\r', '\n')
    
    lines = smiles_text.strip().split('\n')
    
    logger.info(f"parse_smiles_input: received {len(lines)} lines")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        smiles = None
        identifier = None
        
        # Try different separators in order of priority: tab, comma, space
        # Tab is most reliable since SMILES don't contain tabs
        if '\t' in line:
            parts = line.split('\t', 1)
            smiles = parts[0].strip()
            identifier = parts[1].strip() if len(parts) > 1 else None
        elif ',' in line and not line.startswith('['):
            # Comma, but be careful - SMILES can contain commas in atom lists like [C,N]
            # Only split on comma if it's clearly a separator (followed by non-SMILES char)
            # Simple heuristic: if there's a comma followed by a space or letter, it's a separator
            comma_idx = line.rfind(',')  # Use last comma
            after_comma = line[comma_idx+1:].strip() if comma_idx >= 0 else ""
            # Check if after comma looks like an identifier (starts with letter, no SMILES chars)
            if comma_idx > 0 and after_comma and after_comma[0].isalpha() and '=' not in after_comma and '(' not in after_comma:
                smiles = line[:comma_idx].strip()
                identifier = after_comma
            else:
                smiles = line
        elif ' ' in line:
            # Space - split on last space that's followed by identifier-like text
            # SMILES don't usually have spaces, so this is usually safe
            parts = line.rsplit(None, 1)  # Split on last whitespace
            if len(parts) == 2:
                potential_smiles = parts[0].strip()
                potential_id = parts[1].strip()
                # Check if the second part looks like an ID (alphanumeric, not SMILES-like)
                if potential_id and potential_id[0].isalpha() and '=' not in potential_id and '(' not in potential_id:
                    smiles = potential_smiles
                    identifier = potential_id
                else:
                    smiles = line
            else:
                smiles = line
        else:
            smiles = line
        
        # Generate default identifier if none provided
        if not identifier:
            identifier = f"ligand_{i+1:04d}"
        
        # Clean identifier (remove problematic characters)
        identifier = re.sub(r'[^\w\-.]', '_', identifier)
        
        if smiles:
            results.append((smiles, identifier))
            logger.info(f"  Parsed line {i}: ID='{identifier}' SMILES='{smiles[:60]}{'...' if len(smiles) > 60 else ''}'")
    
    logger.info(f"parse_smiles_input: returning {len(results)} SMILES")
    return results


def strip_sdf_properties(sdf_block: str) -> str:
    """
    Remove all properties from an SDF block, keeping only the molecule structure.
    This removes OpenBabel-added properties like Energy before passing to GNINA.
    """
    lines = sdf_block.split('\n')
    result_lines = []
    in_properties = False
    
    for line in lines:
        # Properties section starts after M  END
        if line.strip() == 'M  END':
            result_lines.append(line)
            in_properties = True
            continue
        
        # Skip property lines (start with > or are property values)
        if in_properties:
            if line.strip().startswith('>'):
                continue
            # Skip blank lines and property values in properties section
            # but stop at $$$$ which marks end of molecule
            if line.strip() == '$$$$':
                in_properties = False
                result_lines.append(line)
            continue
        
        result_lines.append(line)
    
    return '\n'.join(result_lines)


def prepare_single_ligand(args: Tuple[str, int, float, str]) -> Tuple[int, Optional[str], Optional[str], str]:
    """
    Prepare a single ligand from SMILES: protonate at pH with OpenBabel, generate 3D, minimize.
    Returns: (index, mol_block or None, error or None, identifier)
    
    This function must be at module level for multiprocessing to work.
    
    Note: OpenBabel's -p and --gen3d flags are incompatible (--gen3d resets protonation).
    So we do it in two steps: 1) protonate to get correct SMILES, 2) generate 3D from protonated SMILES.
    """
    smiles, idx, ph, identifier = args
    
    try:
        import subprocess
        import tempfile
        
        # Create temp directory for this ligand
        tmp_dir = tempfile.mkdtemp(prefix=f"lig_{idx}_")
        
        smi_file = os.path.join(tmp_dir, "input.smi")
        protonated_smi_file = os.path.join(tmp_dir, "protonated.smi")
        sdf_file = os.path.join(tmp_dir, "output.sdf")
        
        # Write input SMILES with identifier as molecule name
        with open(smi_file, 'w') as f:
            f.write(f"{smiles} {identifier}\n")
        
        # Step 1: Protonate at specified pH (output as SMILES to preserve protonation state)
        # -r flag strips salts and small fragments (keeps largest fragment)
        cmd_protonate = [
            'obabel', smi_file,
            '-O', protonated_smi_file,
            '-r',               # Remove salts/small fragments
            '-p', str(ph)
        ]
        
        result1 = subprocess.run(cmd_protonate, capture_output=True, text=True, timeout=30)
        
        if not os.path.exists(protonated_smi_file) or os.path.getsize(protonated_smi_file) == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return idx, None, f"Protonation failed: {result1.stderr[:100] if result1.stderr else 'No output'}", identifier
        
        # Step 2: Generate 3D from protonated SMILES and minimize
        # --gen3d best: use best (slowest) 3D coordinate generation
        # --minimize: energy minimization
        # --ff MMFF94s: MMFF94 with static charges (better for charged molecules)
        # --crit 1e-7: convergence criterion
        # --sd: steepest descent minimization
        cmd_3d = [
            'obabel', protonated_smi_file,
            '-O', sdf_file,
            '--gen3d', 'medium',
            '--minimize',
            '--ff', 'MMFF94s',
            '--crit', '1e-7',
            '--sd'
        ]
        
        result2 = subprocess.run(cmd_3d, capture_output=True, text=True, timeout=120)
        
        # Read output
        if os.path.exists(sdf_file) and os.path.getsize(sdf_file) > 50:
            with open(sdf_file, 'r') as f:
                sdf_block = f.read()
            
            # Cleanup
            shutil.rmtree(tmp_dir, ignore_errors=True)
            
            # Validate SDF block has required components
            if sdf_block and '$$$$' in sdf_block and 'M  END' in sdf_block:
                # Fix the molecule name in the SDF block (first line)
                # OpenBabel may have mangled it, so we replace it with just the identifier
                lines = sdf_block.split('\n')
                if lines:
                    lines[0] = identifier  # Set first line to just the identifier
                sdf_block = '\n'.join(lines)
                
                # Strip OpenBabel properties (like Energy) - GNINA will add its own
                sdf_block = strip_sdf_properties(sdf_block)
                
                return idx, sdf_block, None, identifier
            else:
                return idx, None, f"Incomplete SDF generated (missing M END or $$$$)", identifier
        
        # If that failed, try without minimization
        cmd_3d_simple = [
            'obabel', protonated_smi_file,
            '-O', sdf_file,
            '--gen3d', 'best'
        ]
        
        result3 = subprocess.run(cmd_3d_simple, capture_output=True, text=True, timeout=60)
        
        if os.path.exists(sdf_file) and os.path.getsize(sdf_file) > 50:
            with open(sdf_file, 'r') as f:
                sdf_block = f.read()
            
            shutil.rmtree(tmp_dir, ignore_errors=True)
            
            # Validate SDF block has required components
            if sdf_block and '$$$$' in sdf_block and 'M  END' in sdf_block:
                # Fix the molecule name in the SDF block (first line)
                lines = sdf_block.split('\n')
                if lines:
                    lines[0] = identifier
                sdf_block = '\n'.join(lines)
                
                # Strip OpenBabel properties
                sdf_block = strip_sdf_properties(sdf_block)
                
                return idx, sdf_block, None, identifier
        
        # Cleanup and return error
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return idx, None, f"3D generation failed: {result2.stderr[:100] if result2.stderr else 'Unknown error'}", identifier
        
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return idx, None, f"Timeout processing: {smiles[:50]}", identifier
    except Exception as e:
        if 'tmp_dir' in locals():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return idx, None, f"Error processing {smiles[:50]}...: {str(e)}", identifier


def prepare_ligand_batch(batch_args: List[Tuple[str, int, float, str]]) -> List[Tuple[int, Optional[str], Optional[str], str]]:
    """Process a batch of ligands."""
    results = []
    for args in batch_args:
        results.append(prepare_single_ligand(args))
    return results


# ============================================================================
# GNINA DOCKING ENGINE
# ============================================================================

class GninaDockingEngine:
    """Manages GNINA docking operations with GPU load balancing."""
    
    def __init__(self, gnina_path: str = GNINA_PATH):
        self.gnina_path = gnina_path
        self.gpu_semaphores = [asyncio.Semaphore(1) for _ in range(N_GPU)]
        self._verify_gnina()
    
    def _verify_gnina(self):
        """Verify GNINA binary is available."""
        if not os.path.exists(self.gnina_path):
            # Try to find gnina in PATH
            result = subprocess.run(['which', 'gnina'], capture_output=True, text=True)
            if result.returncode == 0:
                self.gnina_path = result.stdout.strip()
            else:
                logger.warning(f"GNINA not found at {self.gnina_path}. Please set GNINA_PATH.")
                return
        
        try:
            result = subprocess.run(
                [self.gnina_path, '--version'],
                capture_output=True, text=True, timeout=10
            )
            logger.info(f"GNINA version: {result.stdout.strip()[:100]}")
        except Exception as e:
            logger.warning(f"Could not verify GNINA: {e}")
    
    async def dock_batch(
        self,
        receptor_path: str,
        ligand_path: str,
        output_path: str,
        reference_path: str,
        gpu_id: int = 0,
        num_modes: int = 9,
        exhaustiveness: int = 8,
        cnn_scoring: str = 'rescore',
        autobox_add: float = 4.0,
        seed: int = 666
    ) -> Tuple[bool, str]:
        """
        Run GNINA docking on a specific GPU.
        
        Args:
            receptor_path: Path to receptor PDB file
            ligand_path: Path to ligands SDF file
            output_path: Path for output docked poses
            reference_path: Path to reference ligand for autobox
            gpu_id: GPU device ID
            num_modes: Number of binding modes to generate
            exhaustiveness: Search exhaustiveness
            cnn_scoring: CNN scoring mode
            autobox_add: Padding around reference ligand for search box
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        cmd = [
            self.gnina_path,
            '-r', receptor_path,
            '-l', ligand_path,
            '-o', output_path,
            '--autobox_ligand', reference_path,
            '--autobox_add', str(autobox_add),
            '--num_modes', str(num_modes),
            '--exhaustiveness', str(exhaustiveness),
            '--cnn_scoring', cnn_scoring,
            '--cpu', str(CPU_PER_GPU),
            '--seed', str(seed)
        ]

        # GNINA v1.3.2 Torch backend ignores --device; use CUDA_VISIBLE_DEVICES instead
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        async with self.gpu_semaphores[gpu_id % len(self.gpu_semaphores)]:
            logger.info(f"Starting docking on GPU {gpu_id}: {os.path.basename(ligand_path)}")
            logger.info(f"GNINA command: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            stdout, stderr = await proc.communicate()
            
            # Log GNINA output for debugging
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            if stdout_str:
                logger.info(f"GNINA GPU {gpu_id} stdout: {stdout_str[:500]}")
            if stderr_str:
                logger.info(f"GNINA GPU {gpu_id} stderr: {stderr_str[:500]}")
            
            if proc.returncode != 0:
                error_msg = stderr_str[:500]
                logger.error(f"GNINA failed on GPU {gpu_id}: {error_msg}")
                return False, error_msg
            
            # Check output file
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    content = f.read()
                pose_count = content.count('$$$$')
                has_vina = 'minimizedAffinity' in content
                logger.info(f"GNINA GPU {gpu_id} output: {pose_count} poses, has_vina_scores={has_vina}")
            
            logger.info(f"Docking completed on GPU {gpu_id}: {os.path.basename(output_path)}")
            return True, "Success"


# ============================================================================
# JOB PROCESSOR
# ============================================================================

class DockingJobProcessor:
    """Handles the complete docking workflow."""
    
    def __init__(self):
        self.engine = GninaDockingEngine()
        self.process_pool = ProcessPoolExecutor(max_workers=WORKER_CPU)
    
    async def update_progress(self, job_id: str, **kwargs):
        """Update job progress and notify connected WebSockets."""
        if job_id in active_jobs:
            job = active_jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            # Notify WebSocket clients
            if job_id in job_websockets:
                message = {
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "message": job.message,
                    "total_ligands": job.total_ligands,
                    "processed_ligands": job.processed_ligands,
                    "current_stage": job.current_stage,
                    "timings": job.timings
                }
                for ws in job_websockets[job_id][:]:
                    try:
                        await ws.send_json(message)
                    except Exception:
                        job_websockets[job_id].remove(ws)
    
    async def prepare_ligands_from_smiles(
        self,
        job_id: str,
        smiles_with_ids: List[Tuple[str, str]],
        output_path: str,
        ph: float = 7.4
    ) -> Tuple[int, int]:
        """
        Prepare ligands from SMILES strings in parallel.
        
        Args:
            smiles_with_ids: List of (smiles, identifier) tuples
        
        Returns: (success_count, total_count)
        """
        total = len(smiles_with_ids)
        logger.info(f"=" * 60)
        logger.info(f"prepare_ligands_from_smiles: {total} SMILES received")
        for i, (smi, ident) in enumerate(smiles_with_ids):
            logger.info(f"  [{i}] ID='{ident}' SMILES='{smi}'")
        logger.info(f"=" * 60)
        
        await self.update_progress(
            job_id,
            status=JobStatus.PREPARING,
            message=f"Preparing {total} ligands...",
            total_ligands=total,
            current_stage="3D Generation & Minimization"
        )
        
        # Create tasks with indices and identifiers
        tasks = [(smi, i, ph, identifier) for i, (smi, identifier) in enumerate(smiles_with_ids)]

        # Run in a thread pool so multiple ligands are prepared in parallel.
        # prepare_single_ligand calls subprocess (obabel) so it releases the GIL.
        all_results = []
        errors = []
        loop = asyncio.get_event_loop()
        n_workers = min(total, WORKER_CPU)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {loop.run_in_executor(pool, prepare_single_ligand, task): task for task in tasks}
            done_count = 0
            for future in asyncio.as_completed(futures):
                try:
                    res_idx, mol_block, error, res_ident = await future
                    if mol_block:
                        all_results.append((res_idx, mol_block, res_ident))
                    else:
                        logger.warning(f"  FAILED: {res_ident}: {error}")
                        errors.append(f"{res_ident}: {error}")
                except Exception as e:
                    logger.error(f"  EXCEPTION in prepare_single_ligand: {e}")
                    errors.append(str(e))
                done_count += 1
                await self.update_progress(
                    job_id,
                    progress=30 * (done_count / total),
                    processed_ligands=done_count,
                    message=f"Prepared {len(all_results)}/{done_count} ligands ({len(errors)} failed)"
                )
        
        logger.info(f"=" * 60)
        logger.info(f"Processing complete: {len(all_results)} success, {len(errors)} errors")
        for err in errors:
            logger.warning(f"  Error: {err}")
        logger.info(f"=" * 60)
        
        # Sort by index and write to SDF
        all_results.sort(key=lambda x: x[0])
        
        logger.info(f"Writing {len(all_results)} molecules to {output_path}")
        with open(output_path, 'w') as f:
            for idx, mol_block, identifier in all_results:
                logger.info(f"  Writing [{idx}]: {identifier}")
                # Ensure mol_block is clean
                mol_block = mol_block.strip()
                f.write(mol_block)
                # Add newline if needed
                if not mol_block.endswith('\n'):
                    f.write('\n')
                # Ensure each molecule ends with $$$$
                if not mol_block.endswith('$$$$'):
                    f.write('$$$$\n')
        
        # Verify the written file
        with open(output_path, 'r') as f:
            content = f.read()
        mol_count_by_delim = content.count('$$$$')
        logger.info(f"Written file stats: {mol_count_by_delim} $$$$ delimiters, {len(content)} bytes")
        
        # Also log first line of each molecule
        blocks = content.split('$$$$')
        for i, block in enumerate(blocks[:12]):  # First 12
            if block.strip():
                first_line = block.strip().split('\n')[0]
                logger.info(f"  Block [{i}] first line: '{first_line}'")
        
        verify_suppl = Chem.SDMolSupplier(output_path)
        verified_count = sum(1 for mol in verify_suppl if mol is not None)
        logger.info(f"RDKit verification: reads {verified_count} molecules")
        
        if errors:
            logger.warning(f"Ligand preparation errors: {errors}")
        
        return len(all_results), total

    async def generate_pymol_session(
        self,
        work_dir: str,
        receptor_path: str,
        reference_path: str,
        docking_results_sdf: str,
        session_name: str = '',
    ) -> Optional[str]:
        """
        Generate a PyMOL session (.pse) from the docked SDF output.

        Each unique ligand becomes one multi-state PyMOL object (one state per pose),
        so the user can cycle through poses with PyMOL's state controls.
        Pose files are written from the raw SDF text — never through SDWriter —
        to avoid bond-table corruption.

        Returns path to the .pse file, or None on failure.
        """
        PYMOL_EXE = '/home/evehom/Programs/miniconda3/envs/pymol/bin/pymol'

        def _pymol_name(name: str) -> str:
            """Make a valid PyMOL object name."""
            s = re.sub(r'[^\w]', '_', name)
            if s and s[0].isdigit():
                s = 'lig_' + s
            return s or 'ligand'

        pymol_dir = os.path.join(work_dir, 'pymol_files')
        os.makedirs(pymol_dir, exist_ok=True)
        pse_filename = f'{session_name}.pse' if session_name else 'visualization.pse'
        pse_path = os.path.join(work_dir, pse_filename)

        # Read output SDF and group blocks by molecule name (first line)
        with open(docking_results_sdf, 'r') as f:
            content = f.read()
        raw_blocks = [b for b in content.split('$$$$') if b.strip()]

        from collections import defaultdict
        ligand_blocks: Dict[str, list] = defaultdict(list)
        for block in raw_blocks:
            mol_name = block.strip().split('\n')[0].strip() or 'unknown'
            ligand_blocks[mol_name].append(block.strip())

        # Write one SDF per ligand (all poses → multi-state object in PyMOL)
        ligand_entries = []  # (pymol_obj_name, sdf_path)
        for mol_name, blocks in ligand_blocks.items():
            obj_name = _pymol_name(mol_name)
            lig_sdf = os.path.join(pymol_dir, f'{obj_name}.sdf')
            with open(lig_sdf, 'w') as f:
                for b in blocks:
                    f.write(b.rstrip('\r\n ') + '\n$$$$\n')
            ligand_entries.append((obj_name, lig_sdf))

        # Build PyMOL Python script
        load_cmds = '\n'.join(
            f"cmd.load(r'{sdf}', '{obj}')\n"
            f"cmd.show('sticks', '{obj}')\n"
            f"cmd.hide('nonbonded', '{obj}')"
            for obj, sdf in ligand_entries
        )
        ligand_obj_list = ', '.join(f"'{obj}'" for obj, _ in ligand_entries)
        script = f"""from pymol import cmd

# --- Protein ---
cmd.load(r'{receptor_path}', 'protein')
cmd.hide('everything', 'protein')
cmd.show('cartoon', 'protein')
cmd.spectrum('count', 'rainbow', 'protein and name CA')

# --- Reference ligand: green carbons, element colours, polar H shown ---
cmd.load(r'{reference_path}', 'reference_ligand')
cmd.show('sticks', 'reference_ligand')
cmd.hide('nonbonded', 'reference_ligand')
cmd.util.cbag('reference_ligand')
cmd.hide('sticks', 'reference_ligand and hydro and (neighbor (elem C))')

# --- Docked poses: polar H shown, non-polar H hidden ---
{load_cmds}

ligand_objs = [{ligand_obj_list}]
for obj in ligand_objs:
    cmd.hide('sticks', obj + ' and hydro and (neighbor (elem C))')

# --- Binding-site residues within 5 Å of any docked pose: polar H shown ---
all_ligands_sel = ' or '.join(ligand_objs) if ligand_objs else 'none'
cmd.select('binding_site', f'protein and byres (protein within 5 of ({{all_ligands_sel}}))')
cmd.show('lines', 'binding_site')
cmd.hide('lines', 'binding_site and hydro and (neighbor (elem C))')

# Transparent light-grey surface on binding-site residues only, on the protein object
# surface_color overrides surface colour independently of cartoon colour
cmd.show('surface', 'binding_site')
cmd.set('surface_color', 'grey90', 'protein')
cmd.set('transparency', 0.5, 'protein')

cmd.deselect()
cmd.orient(all_ligands_sel if ligand_objs else 'protein')
cmd.zoom('binding_site', 5)
cmd.save(r'{pse_path}')
cmd.quit()
"""
        script_path = os.path.join(pymol_dir, 'make_session.py')
        with open(script_path, 'w') as f:
            f.write(script)

        try:
            proc = await asyncio.create_subprocess_exec(
                PYMOL_EXE, '-cq', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            if proc.returncode != 0:
                logger.error(f"PyMOL exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}")
                return None
            if not os.path.exists(pse_path):
                logger.error(f"PyMOL ran but {pse_path} was not created")
                return None
            logger.info(f"PyMOL session: {pse_path} ({os.path.getsize(pse_path):,} bytes)")
            return pse_path
        except asyncio.TimeoutError:
            logger.error("PyMOL timed out after 300 s")
            return None
        except Exception as e:
            logger.error(f"PyMOL session generation failed: {e}")
            return None

    async def run_docking_job(
        self,
        job_id: str,
        receptor_path: str,
        reference_path: str,
        ligand_path: str,
        output_dir: str,
        num_poses: int = 9,
        exhaustiveness: int = 8,
        cnn_scoring: str = 'rescore',
        seed: int = 666
    ) -> str:
        """
        Run docking job with GPU load balancing.
        
        Returns: Path to merged results file
        """
        await self.update_progress(
            job_id,
            status=JobStatus.DOCKING,
            progress=35,
            message="Starting molecular docking...",
            current_stage="GPU Docking"
        )
        
        # Count ligands by splitting SDF text (more reliable than RDKit)
        with open(ligand_path, 'r') as f:
            sdf_content = f.read()
        
        # Split on $$$$ to get individual molecules
        mol_blocks = sdf_content.split('$$$$')
        # Filter out empty blocks and re-add the $$$$ delimiter
        mol_blocks = [block.strip() + '\n$$$$\n' for block in mol_blocks if block.strip()]
        total_mols = len(mol_blocks)
        
        logger.info(f"Found {total_mols} molecules in ligand file")
        
        if total_mols == 0:
            raise ValueError("No valid molecules in ligand file")
        
        # For small numbers of ligands, just use one GPU
        if total_mols <= 5:
            num_gpus_to_use = 1
        else:
            num_gpus_to_use = min(N_GPU, total_mols)
        
        # Split ligands evenly across GPUs
        mols_per_gpu = (total_mols + num_gpus_to_use - 1) // num_gpus_to_use
        
        gpu_tasks = []
        output_files = []
        
        for i in range(num_gpus_to_use):
            gpu_id = DOCK_GPU_ID + i
            start_idx = i * mols_per_gpu
            end_idx = min(start_idx + mols_per_gpu, total_mols)
            
            if start_idx >= total_mols:
                break
            
            batch_blocks = mol_blocks[start_idx:end_idx]
            
            # Write batch to file (text-based, preserving original format)
            batch_path = os.path.join(output_dir, f"ligands_gpu{gpu_id}.sdf")
            output_path = os.path.join(output_dir, f"docked_gpu{gpu_id}.sdf")
            
            with open(batch_path, 'w') as f:
                f.write(''.join(batch_blocks))
            
            logger.info(f"GPU {gpu_id}: {len(batch_blocks)} ligands written to {batch_path}")
            
            # Create docking task
            task = self.engine.dock_batch(
                receptor_path=receptor_path,
                ligand_path=batch_path,
                output_path=output_path,
                reference_path=reference_path,
                gpu_id=gpu_id,
                num_modes=num_poses,
                exhaustiveness=exhaustiveness,
                cnn_scoring=cnn_scoring,
                seed=seed
            )
            gpu_tasks.append(task)
            output_files.append(output_path)
        
        await self.update_progress(
            job_id,
            message=f"Docking {total_mols} ligands on {len(gpu_tasks)} GPU(s)..."
        )
        
        # Run all GPU tasks in parallel
        results = await asyncio.gather(*gpu_tasks, return_exceptions=True)
        
        # Check for errors
        for i, result in enumerate(results):
            physical_gpu = DOCK_GPU_ID + i
            if isinstance(result, Exception):
                logger.error(f"GPU {physical_gpu} task failed: {result}")
            elif not result[0]:
                logger.error(f"GPU {physical_gpu} docking failed: {result[1]}")
        
        await self.update_progress(
            job_id,
            progress=85,
            message="Merging results..."
        )
        
        # Merge output files
        merged_path = os.path.join(output_dir, "docked_merged.sdf")
        total_poses = 0
        with open(merged_path, 'w') as outfile:
            for out_path in output_files:
                if os.path.exists(out_path):
                    with open(out_path, 'r') as infile:
                        content = infile.read()
                        poses_in_file = content.count('$$$$')
                        logger.info(f"Merging {out_path}: {poses_in_file} poses")
                        total_poses += poses_in_file
                        outfile.write(content)
                else:
                    logger.warning(f"Output file not found: {out_path}")
        
        logger.info(f"Merged total: {total_poses} poses")
        
        return merged_path
    
    def sort_and_filter_results(
        self,
        input_path: str,
        output_path: str,
        sort_by: str = 'minimizedAffinity',
        max_poses: Optional[int] = None,
    ) -> int:
        """
        Sort docked poses by score and optionally limit output.
        
        Score types:
        - minimizedAffinity: Vina score (kcal/mol), lower is better
        - CNNaffinity: CNN predicted affinity, higher is better  
        - CNN_VS: CNNaffinity corrected by CNNscore, higher is better
        
        Returns: Number of poses in output
        """
        # Guard: empty merged SDF means all GPU tasks failed
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            logger.error(f"sort_and_filter_results: input file is empty or missing: {input_path}")
            return 0

        # Pre-process GNINA output: merge split structure/properties blocks before RDKit parsing
        with open(input_path, 'r', errors='replace') as _f:
            raw_content = _f.read()
        if not raw_content.strip():
            logger.error(f"sort_and_filter_results: input file has no content: {input_path}")
            return 0
        raw_block_count = raw_content.count('$$$$')
        logger.info(f"sort_and_filter_results: raw GNINA output has {raw_block_count} $$$$ blocks")
        fixed_content = _fix_split_sdf_blocks(raw_content)
        fixed_block_count = fixed_content.count('$$$$')
        logger.info(f"sort_and_filter_results: after split-block fix: {fixed_block_count} blocks "
                    f"({'merged ' + str(raw_block_count - fixed_block_count) + ' split pairs' if raw_block_count != fixed_block_count else 'no merging needed'})")
        fixed_path = input_path + ".fixed.sdf"
        with open(fixed_path, 'w') as _f:
            _f.write(fixed_content)

        # Verify what was written matches expectations
        with open(fixed_path, 'r') as _f:
            _disk = _f.read()
        logger.info(f"sort_and_filter_results: fixed file on disk: {_disk.count('$$$$')} $$$$ terminators, {len(_disk)} bytes")
        # Log first block for format inspection
        _first_sep = _disk.find('$$$$')
        if _first_sep >= 0:
            logger.info(f"sort_and_filter_results: first block (repr, last 200 chars before $$$$): {repr(_disk[max(0,_first_sep-200):_first_sep+4])}")

        # Split fixed content into raw blocks (one per molecule) for pose file writing
        # This preserves exact GNINA output format, avoiding SDWriter aromaticity issues.
        # After the split-block fix all blocks should be merged, so len(raw_blocks) == len(suppl).
        raw_blocks = [b for b in fixed_content.split('$$$$') if b.strip()]
        logger.info(f"sort_and_filter_results: {len(raw_blocks)} raw blocks after merging")

        # Use RDKit SDMolSupplier for robust parsing of all SDF variants
        # (V2000, V3000, Marvin-style 2D, missing headers, etc.)
        suppl = Chem.SDMolSupplier(fixed_path, removeHs=False, sanitize=False)
        logger.info(f"sort_and_filter_results: SDMolSupplier sees {len(suppl)} entries")
        poses = []
        skipped_none = 0
        skipped_zero_atom = 0
        for i, mol in enumerate(suppl):
            if mol is None:
                skipped_none += 1
                raw_name = raw_blocks[i].strip().split('\n')[0] if i < len(raw_blocks) else '?'
                logger.warning(f"sort_and_filter_results: mol[{i}] ('{raw_name}') returned None from SDMolSupplier — skipped")
                continue
            if mol.GetNumAtoms() == 0:
                skipped_zero_atom += 1
                raw_name = raw_blocks[i].strip().split('\n')[0] if i < len(raw_blocks) else '?'
                logger.warning(f"sort_and_filter_results: mol[{i}] ('{raw_name}') has 0 atoms — skipped")
                continue

            mol_name = _extract_mol_name(mol, f"mol_{i}")
            raw_block = raw_blocks[i] if i < len(raw_blocks) else None

            # Log property names for the first molecule to help diagnose RDKit/GNINA spacing issues
            if i == 0:
                try:
                    logger.info(f"sort_and_filter_results: mol[0] RDKit props: {list(mol.GetPropNames())}")
                except Exception:
                    pass

            # Extract docking score — try RDKit first, then regex on raw block.
            # GNINA writes '>  <minimizedAffinity>' (two spaces); some RDKit builds store
            # it under the trimmed key, others may not parse it at all.
            score = 0.0
            score_found = False
            prop_candidates = ([sort_by] if sort_by != 'minimizedAffinity' else []) + ['minimizedAffinity']
            for prop_name in prop_candidates:
                if mol.HasProp(prop_name):
                    try:
                        score = float(mol.GetProp(prop_name))
                        score_found = True
                        break
                    except (ValueError, KeyError):
                        pass

            # Fallback: parse score directly from raw block text (handles two-space headers)
            if not score_found and raw_block is not None:
                for prop_name in prop_candidates:
                    m = re.search(
                        r'>\s*<' + re.escape(prop_name) + r'>\s*\n\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)',
                        raw_block
                    )
                    if m:
                        try:
                            score = float(m.group(1))
                            score_found = True
                            break
                        except ValueError:
                            pass

            # Fallback to Energy (OpenBabel-prepared ligands)
            if not score_found and mol.HasProp('Energy'):
                try:
                    score = float(mol.GetProp('Energy'))
                except (ValueError, KeyError):
                    pass

            poses.append((score, mol, mol_name, score_found, raw_block))

        if skipped_none:
            logger.warning(f"sort_and_filter_results: {skipped_none} molecules returned None from SDMolSupplier")
        if skipped_zero_atom:
            logger.warning(f"sort_and_filter_results: {skipped_zero_atom} 0-atom entries remain after split-block fix "
                           f"— split-block merging may be incomplete")
        logger.info(f"sort_and_filter_results: read {len(poses)} valid molecules from {input_path}")

        # Sort: minimizedAffinity/minimizedRMSD → lower is better
        #       CNNscore/CNNaffinity/CNN_VS → higher is better
        reverse = sort_by in ['CNNscore', 'CNNaffinity', 'CNN_VS']
        poses.sort(key=lambda x: x[0], reverse=reverse)

        docked_count = sum(1 for p in poses if p[3])
        logger.info(f"Sorted {len(poses)} poses by {sort_by} (reverse={reverse}), {docked_count} have docking scores")

        if max_poses:
            poses = poses[:max_poses]

        # Write combined SDF from raw GNINA blocks (preserves exact format).
        with open(output_path, 'w') as combined_f:
            for rank, (score, mol, mol_name, has_score, raw_block) in enumerate(poses):
                rank_1 = rank + 1

                if raw_block is not None:
                    # strip() removes both leading \n (from \n$$$$\n join) and trailing whitespace.
                    # Leading \n would shift the V2000 header lines and corrupt the structure.
                    base = raw_block.strip()
                    # Add Structure_ID if not already present in the GNINA output block.
                    # Use \n\n so there is a blank line before the field tag (SDF spec).
                    if '> <Structure_ID>' not in base and '> <Structure ID>' not in base:
                        base += f'\n\n> <Structure_ID>\n{mol_name}\n'
                    # Re-strip so DockingRank always gets a proper blank line before it.
                    block_text = base.rstrip('\r\n ') + f'\n\n> <DockingRank>\n{rank_1}\n\n$$$$\n'
                else:
                    # Fallback through RDKit (shouldn't happen for GNINA output)
                    mol.SetIntProp('DockingRank', rank_1)
                    block_text = Chem.MolToMolBlock(mol)
                    for pname in mol.GetPropNames():
                        block_text += f'> <{pname}>\n{mol.GetProp(pname)}\n\n'
                    block_text += f'> <Structure_ID>\n{mol_name}\n\n$$$$\n'

                combined_f.write(block_text)

        logger.info(f"Wrote {len(poses)} poses to {output_path}")
        return len(poses)

    def add_mcs_rmsd(self, sdf_path: str, reference_path: str) -> int:
        """
        Annotates each pose in sdf_path with an MCS_RMSD field (in-place).

        For each pose the Maximum Common Substructure (MCS) with the reference
        ligand is found; RMSD is then computed on that atom subset. Poses where
        MCS has < 3 atoms or matching fails are written as 'N/A'.

        Returns the number of poses successfully annotated with a numeric RMSD.
        """
        # Load reference ligand — strip H for heavy-atom-only MCS/RMSD.
        # H positions vary with pH correction and crystal packing; including them
        # shrinks the MCS and inflates RMSD in a chemically uninformative way.
        ref_suppl = Chem.SDMolSupplier(reference_path, removeHs=True, sanitize=False)
        ref_mol = None
        for m in ref_suppl:
            if m is not None and m.GetNumAtoms() > 0:
                ref_mol = m
                break
        if ref_mol is None:
            logger.error("add_mcs_rmsd: could not load reference ligand from %s", reference_path)
            return 0
        try:
            Chem.SanitizeMol(ref_mol)
        except Exception:
            pass

        with open(sdf_path, 'r') as f:
            content = f.read()
        blocks = [b for b in content.split('$$$$') if b.strip()]

        annotated = 0
        new_blocks = []

        for block in blocks:
            try:
                pose_mol = Chem.MolFromMolBlock(block.strip(), removeHs=True, sanitize=False)
                if pose_mol is None or pose_mol.GetNumAtoms() == 0:
                    raise ValueError("empty mol")
                try:
                    Chem.SanitizeMol(pose_mol)
                except Exception:
                    pass

                mcs_result = rdFMCS.FindMCS(
                    [ref_mol, pose_mol],
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareOrder,
                    ringMatchesRingOnly=True,
                    completeRingsOnly=False,
                    timeout=5,
                )

                if mcs_result.numAtoms < 3:
                    rmsd_str = 'N/A'
                else:
                    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
                    pose_match = pose_mol.GetSubstructMatch(mcs_mol)

                    if not ref_match or not pose_match:
                        rmsd_str = 'N/A'
                    else:
                        ref_conf = ref_mol.GetConformer()
                        pose_conf = pose_mol.GetConformer()
                        ref_coords = np.array([list(ref_conf.GetAtomPosition(i)) for i in ref_match])
                        pose_coords = np.array([list(pose_conf.GetAtomPosition(i)) for i in pose_match])
                        rmsd = float(np.sqrt(((ref_coords - pose_coords) ** 2).sum(axis=1).mean()))
                        rmsd_str = f'{rmsd:.4f}'
                        annotated += 1

            except Exception as e:
                logger.warning("add_mcs_rmsd: pose annotation failed: %s", e)
                rmsd_str = 'N/A'

            # strip() removes leading \n (blocks 2–N start with \n after splitting on $$$$)
            # which would shift the V2000 header and corrupt the structure.
            base = block.strip()
            base += f'\n\n> <MCS_RMSD>\n{rmsd_str}\n'
            new_blocks.append(base)

        with open(sdf_path, 'w') as f:
            for b in new_blocks:
                f.write(b.rstrip('\r\n ') + '\n\n$$$$\n')

        logger.info("add_mcs_rmsd: annotated %d/%d poses in %s", annotated, len(blocks), sdf_path)
        return annotated

    def add_shape_sim(self, sdf_path: str, reference_path: str) -> int:
        """
        Annotates each pose in sdf_path with a Shape_Sim field (in-place).

        Shape Tanimoto similarity (0–1, higher = more similar) is computed between
        each pose and the reference ligand using their existing 3D coordinates —
        no alignment needed since GNINA already places poses in the binding site.
        Heavy atoms only; poses where the conformer is missing are written as 'N/A'.

        Returns the number of poses successfully annotated.
        """
        ref_suppl = Chem.SDMolSupplier(reference_path, removeHs=True, sanitize=False)
        ref_mol = None
        for m in ref_suppl:
            if m is not None and m.GetNumAtoms() > 0:
                ref_mol = m
                break
        if ref_mol is None:
            logger.error("add_shape_sim: could not load reference ligand from %s", reference_path)
            return 0
        try:
            Chem.SanitizeMol(ref_mol)
        except Exception:
            pass
        if not ref_mol.GetNumConformers():
            logger.error("add_shape_sim: reference ligand has no 3D conformer")
            return 0

        with open(sdf_path, 'r') as f:
            content = f.read()
        blocks = [b for b in content.split('$$$$') if b.strip()]

        annotated = 0
        new_blocks = []

        for block in blocks:
            try:
                pose_mol = Chem.MolFromMolBlock(block.strip(), removeHs=True, sanitize=False)
                if pose_mol is None or pose_mol.GetNumAtoms() == 0:
                    raise ValueError("empty mol")
                if not pose_mol.GetNumConformers():
                    raise ValueError("no conformer")
                try:
                    Chem.SanitizeMol(pose_mol)
                except Exception:
                    pass

                dist = rdShapeHelpers.ShapeTanimotoDist(ref_mol, pose_mol)
                sim = round(1.0 - dist, 4)
                sim_str = f'{sim:.4f}'
                annotated += 1

            except Exception as e:
                logger.warning("add_shape_sim: pose annotation failed: %s", e)
                sim_str = 'N/A'

            base = block.strip()
            base += f'\n\n> <Shape_Sim>\n{sim_str}\n'
            new_blocks.append(base)

        with open(sdf_path, 'w') as f:
            for b in new_blocks:
                f.write(b.rstrip('\r\n ') + '\n\n$$$$\n')

        logger.info("add_shape_sim: annotated %d/%d poses in %s", annotated, len(blocks), sdf_path)
        return annotated

    def add_ref_sim(self, sdf_path: str, reference_path: str) -> int:
        """
        Annotates each pose in sdf_path with Ref_Sim — Morgan ECFP4 (radius=2, 2048 bits)
        Tanimoto similarity to the reference ligand (2D structure, heavy atoms only).
        """
        ref_suppl = Chem.SDMolSupplier(reference_path, removeHs=True, sanitize=False)
        ref_mol = None
        for m in ref_suppl:
            if m is not None and m.GetNumAtoms() > 0:
                ref_mol = m
                break
        if ref_mol is None:
            logger.error("add_ref_sim: could not load reference ligand from %s", reference_path)
            return 0
        try:
            Chem.SanitizeMol(ref_mol)
        except Exception:
            pass
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, radius=2, nBits=2048)

        with open(sdf_path, 'r') as f:
            content = f.read()
        blocks = [b for b in content.split('$$$$') if b.strip()]

        annotated = 0
        new_blocks = []

        for block in blocks:
            try:
                pose_mol = Chem.MolFromMolBlock(block.strip(), removeHs=True, sanitize=False)
                if pose_mol is None or pose_mol.GetNumAtoms() == 0:
                    raise ValueError("empty mol")
                try:
                    Chem.SanitizeMol(pose_mol)
                except Exception:
                    pass
                pose_fp = AllChem.GetMorganFingerprintAsBitVect(pose_mol, radius=2, nBits=2048)
                sim = DataStructs.TanimotoSimilarity(ref_fp, pose_fp)
                sim_str = f'{sim:.4f}'
                annotated += 1
            except Exception as e:
                logger.warning("add_ref_sim: pose annotation failed: %s", e)
                sim_str = 'N/A'

            base = block.strip()
            base += f'\n\n> <Ref_Sim>\n{sim_str}\n'
            new_blocks.append(base)

        with open(sdf_path, 'w') as f:
            for b in new_blocks:
                f.write(b.rstrip('\r\n ') + '\n\n$$$$\n')

        logger.info("add_ref_sim: annotated %d/%d poses in %s", annotated, len(blocks), sdf_path)
        return annotated

    def add_posebusters_flags(self, sdf_path: str) -> int:
        """
        Annotates each pose in sdf_path with PB_failures — the count of PoseBusters
        checks that failed (0 = all pass). Runs with config='mol' which covers
        geometry checks (bond lengths, angles, planarity, internal clashes) without
        requiring the receptor or reference ligand.
        Modifies sdf_path in-place. Returns number of poses successfully evaluated.
        """
        try:
            from posebusters import PoseBusters
        except ImportError:
            logger.error("add_posebusters_flags: posebusters is not installed")
            return 0

        with open(sdf_path, 'r') as f:
            content = f.read()
        blocks = [b for b in content.split('$$$$') if b.strip()]

        try:
            pb = PoseBusters(config='mol')
            df = pb.bust(sdf_path, full_report=True)
        except Exception as e:
            logger.error("add_posebusters_flags: PoseBusters failed: %s", e)
            return 0

        # Count failed checks per row (False = fail, NaN = not applicable → ignore)
        _INFRA_COLS = {'mol_cond_loaded', 'mol_true_loaded'}
        bool_cols = [c for c in df.columns if df[c].dtype == bool and c not in _INFRA_COLS]
        fail_counts = []
        for _, row in df.iterrows():
            vals = row[bool_cols].dropna()
            fail_counts.append(int((vals == False).sum()))  # noqa: E712

        annotated = 0
        new_blocks = []
        for i, block in enumerate(blocks):
            if i < len(fail_counts):
                count_str = str(fail_counts[i])
                annotated += 1
            else:
                count_str = 'N/A'
            base = block.strip()
            base += f'\n\n> <PB_Flags>\n{count_str}\n'
            new_blocks.append(base)

        with open(sdf_path, 'w') as f:
            for b in new_blocks:
                f.write(b.rstrip('\r\n ') + '\n\n$$$$\n')

        # Log which checks fail most frequently to help diagnose systematic issues
        if not df.empty:
            fail_totals = (df[bool_cols] == False).sum().sort_values(ascending=False)  # noqa: E712
            failing = fail_totals[fail_totals > 0]
            if not failing.empty:
                logger.info("add_posebusters_flags: most common failures across all poses:\n%s",
                            failing.to_string())

        logger.info("add_posebusters_flags: evaluated %d/%d poses in %s",
                    annotated, len(blocks), sdf_path)
        return annotated


# ============================================================================
# PYMOL SCRIPT GENERATION
# ============================================================================

def generate_pymol_script(
    receptor_filename: str,
    reference_filename: str,
    pose_metadata: List[Dict[str, Any]]
) -> str:
    """
    Generate a PyMOL .pml visualization script for docking results.

    The ZIP must be extracted to a directory and PyMOL run from there:
        pymol visualization.pml
    """
    lines = [
        "# GNINA Molecular Docking - PyMOL Visualization Script",
        "# Generated by GNINA Docking Web App",
        "# Usage: extract the ZIP, then run: pymol visualization.pml",
        "",
        "# Load structures",
        f"load receptor/{receptor_filename}, protein",
        f"load reference/{reference_filename}, reference_ligand",
        "",
        "# Load docking poses as individual objects",
    ]

    obj_names = []
    for p in pose_metadata:
        lines.append(f"load poses/{p['sdf_filename']}, {p['obj_name']}")
        obj_names.append(p['obj_name'])

    lines += [
        "",
        "# --- Protein: rainbow cartoon ---",
        "hide everything, protein",
        "show cartoon, protein",
        "spectrum count, rainbow, protein",
        "",
        "# --- Reference ligand: yellow sticks ---",
        "hide everything, reference_ligand",
        "show sticks, reference_ligand",
        "util.cbaw reference_ligand",
        "color yellow, (reference_ligand and elem C)",
        "",
        "# --- Docking poses: sticks with green carbons ---",
    ]

    for obj_name in obj_names:
        lines.append(f"show sticks, {obj_name}")
        lines.append(f"util.cbag {obj_name}")

    all_poses_sel = " or ".join(obj_names) if obj_names else "none"

    lines += [
        "",
        "# --- Binding site: residues within 5 Å of any docking pose ---",
        f"select pose_atoms, ({all_poses_sel})",
        "select binding_residues, byres (protein within 5 of pose_atoms)",
        "",
        "# Lines with element colors (white carbons, CPK for N/O/S/etc.)",
        "show lines, binding_residues",
        "util.cbaw binding_residues",
        "",
        "# Transparent gray surface on binding site residues",
        "create binding_site_surface, binding_residues",
        "hide everything, binding_site_surface",
        "show surface, binding_site_surface",
        "color gray70, binding_site_surface",
        "set transparency, 0.5, binding_site_surface",
        "",
        "# --- Final view ---",
        "deselect",
        "zoom protein",
        "set ray_shadows, 0",
        "bg_color white",
    ]

    return '\n'.join(lines) + '\n'


def generate_pymol_session(
    receptor_path: str,
    reference_path: str,
    pose_metadata: List[Dict[str, Any]],
    poses_dir: Path,
    output_pse: str,
    pymol_path: str = PYMOL_PATH
) -> bool:
    """
    Generate a self-contained PyMOL session file (.pse) by running PyMOL headlessly.
    All molecular data is embedded in the .pse — no external files needed to open it.
    Returns True on success, False if PyMOL is unavailable or fails (caller falls back to .pml).
    """
    if not os.path.exists(pymol_path):
        logger.warning(f"PyMOL not found at {pymol_path}, falling back to .pml")
        return False

    obj_names = [p['obj_name'] for p in pose_metadata]
    all_poses_sel = ' or '.join(obj_names) if obj_names else 'none'

    # Escape paths for use in Python string literals inside the generated script
    def esc(p: str) -> str:
        return p.replace('\\', '/')

    script_lines = [
        "from pymol import cmd, util",
        "cmd.reinitialize()",
        "",
        "# Load structures",
        f"cmd.load('{esc(receptor_path)}', 'protein')",
        f"cmd.load('{esc(reference_path)}', 'reference_ligand')",
        "",
        "# Load docking poses",
    ]

    for p in pose_metadata:
        sdf_path = esc(str(poses_dir / p['sdf_filename']))
        script_lines.append(f"cmd.load('{sdf_path}', '{p['obj_name']}')")

    script_lines += [
        "",
        "# Protein: rainbow cartoon",
        "cmd.hide('everything', 'protein')",
        "cmd.show('cartoon', 'protein')",
        "cmd.spectrum('count', 'rainbow', 'protein')",
        "",
        "# Reference ligand: yellow sticks",
        "cmd.hide('everything', 'reference_ligand')",
        "cmd.show('sticks', 'reference_ligand')",
        "util.cbaw('reference_ligand')",
        "cmd.color('yellow', 'reference_ligand and elem C')",
        "",
        "# Docking poses: sticks with green carbons",
    ]

    for obj_name in obj_names:
        script_lines.append(f"cmd.show('sticks', '{obj_name}')")
        script_lines.append(f"util.cbag('{obj_name}')")

    script_lines += [
        "",
        "# Binding site residues within 5 Angstrom of any pose",
        f"cmd.select('pose_atoms', '({all_poses_sel})')",
        "cmd.select('binding_residues', 'byres (protein within 5 of pose_atoms)')",
        "cmd.show('lines', 'binding_residues')",
        "util.cbaw('binding_residues')",
        "",
        "# Transparent gray surface on binding site",
        "cmd.create('binding_site_surface', 'binding_residues')",
        "cmd.hide('everything', 'binding_site_surface')",
        "cmd.show('surface', 'binding_site_surface')",
        "cmd.color('gray70', 'binding_site_surface')",
        "cmd.set('transparency', 0.5, 'binding_site_surface')",
        "",
        "# Final view",
        "cmd.deselect()",
        "cmd.zoom('protein')",
        "cmd.set('ray_shadows', 0)",
        "cmd.bg_color('white')",
        "",
        f"cmd.save('{esc(output_pse)}')",
    ]

    script_content = '\n'.join(script_lines)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        result = subprocess.run(
            [pymol_path, '-cq', script_path],
            capture_output=True, text=True, timeout=180
        )

        if result.stdout:
            logger.info(f"PyMOL stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.info(f"PyMOL stderr: {result.stderr[:500]}")

        if result.returncode != 0:
            logger.error(f"PyMOL failed (rc={result.returncode})")
            return False

        if not os.path.exists(output_pse) or os.path.getsize(output_pse) == 0:
            logger.error(f"PyMOL ran but .pse was not created (script: {script_path})")
            # Log the generated script for debugging
            logger.error(f"PyMOL script content:\n{script_content[:2000]}")
            return False

        logger.info(f"PyMOL session saved: {output_pse} ({os.path.getsize(output_pse)//1024} KB)")
        return True

    except subprocess.TimeoutExpired:
        logger.error("PyMOL session generation timed out")
        return False
    except Exception as e:
        logger.error(f"PyMOL session generation error: {e}")
        return False
    finally:
        os.unlink(script_path)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting GNINA Docking Server")
    logger.info(f"Configuration: {N_CPU} CPUs, {N_GPU} GPUs, {WORKER_CPU} workers")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="GNINA Molecular Docking",
    description="High-performance molecular docking with GPU acceleration",
    version="2.0.0",
    lifespan=lifespan
)

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse as _JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"422 Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return _JSONResponse(status_code=422, content={"detail": exc.errors()})

# Initialize processor
job_processor = DockingJobProcessor()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface."""
    # Try multiple possible locations for the template
    possible_paths = [
        Path(__file__).parent / "templates" / "index.html",
        Path.cwd() / "templates" / "index.html",
        Path("/app/templates/index.html"),
        Path("./templates/index.html"),
    ]
    
    for html_path in possible_paths:
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
    
    # Fallback: return embedded HTML
    return HTMLResponse(content=EMBEDDED_HTML)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "cpus": N_CPU,
        "gpus": N_GPU,
        "workers": WORKER_CPU,
        "gnina_path": GNINA_PATH
    }


@app.post("/dock")
async def dock_molecules(
    receptor: UploadFile = File(..., description="Protein structure in PDB format"),
    reference: UploadFile = File(..., description="Reference ligand for binding site (SDF)"),
    ligand_file: Optional[UploadFile] = File(None, description="Ligands to dock (SDF)"),
    ligand_smiles: Optional[str] = Form(None, description="SMILES strings, one per line"),
    ph: float = Form(7.4, ge=0, le=14, description="pH for protonation"),
    num_poses: int = Form(9, ge=1, le=50, description="Number of poses per ligand"),
    exhaustiveness: int = Form(8, ge=1, le=64, description="Search exhaustiveness"),
    cnn_scoring: Literal['rescore', 'refine', 'score_only', 'none'] = Form('rescore'),
    seed: int = Form(666),
    sort_by: Literal['minimizedAffinity', 'CNNscore', 'CNNaffinity', 'CNN_VS'] = Form('minimizedAffinity'),
    generate_pymol: bool = Form(False),
    mcs_rmsd: bool = Form(False),
    shape_sim: bool = Form(False),
    ref_sim: bool = Form(False),
    posebusters: bool = Form(False),
    session_name: str = Form(''),
):
    """
    Perform molecular docking with GNINA.
    
    Accepts either an SDF file with ligands or a list of SMILES strings.
    The reference ligand is used to define the binding site (autobox).
    """
    # Sanitize session name for use in filenames
    session_name = re.sub(r'[^\w\-]', '_', session_name.strip()).strip('_')

    # Validate inputs
    if not receptor.filename:
        raise HTTPException(400, "Receptor PDB file is required")
    
    if not reference.filename:
        raise HTTPException(400, "Reference ligand SDF is required for binding site definition")
    
    has_file = ligand_file is not None and ligand_file.filename
    has_smiles = ligand_smiles is not None and ligand_smiles.strip()
    
    if not has_file and not has_smiles:
        raise HTTPException(400, "Provide either ligand_file (SDF) or ligand_smiles")
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    active_jobs[job_id] = JobProgress(job_id=job_id)
    
    # Create working directory
    work_dir = WORK_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    try:
        # Save receptor
        receptor_path = work_dir / secure_filename(receptor.filename)
        content = await receptor.read()
        receptor_path.write_bytes(content)
        
        # Save reference
        reference_path = work_dir / secure_filename(reference.filename)
        content = await reference.read()
        reference_path.write_bytes(content)
        
        # Prepare ligands
        ligand_path = work_dir / "ligands_prepared.sdf"
        
        if has_smiles:
            # Parse SMILES with optional identifiers
            smiles_with_ids = parse_smiles_input(ligand_smiles)
            
            if not smiles_with_ids:
                raise HTTPException(400, "No valid SMILES strings found")
            
            if len(smiles_with_ids) > 10000:
                raise HTTPException(400, f"Too many ligands ({len(smiles_with_ids)}). Maximum is 10,000.")
            
            prep_start = datetime.now()
            success_count, total_count = await job_processor.prepare_ligands_from_smiles(
                job_id=job_id,
                smiles_with_ids=smiles_with_ids,
                output_path=str(ligand_path),
                ph=ph
            )
            active_jobs[job_id].timings['preparation'] = (datetime.now() - prep_start).total_seconds()
            
            if success_count == 0:
                # Log all errors for debugging
                logger.error(f"All ligand preparations failed.")
                raise HTTPException(400, f"Failed to prepare any ligands from SMILES.")
            
            logger.info(f"Prepared {success_count}/{total_count} ligands")
        
        else:
            # Save uploaded SDF to raw path
            content = await ligand_file.read()
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = content.decode('latin-1')

            raw_sdf_path = work_dir / "ligands_uploaded_raw.sdf"
            raw_sdf_path.write_text(content_str)

            # Parse: separate 3D-ready molecules from 2D molecules needing 3D generation
            suppl = Chem.SDMolSupplier(str(raw_sdf_path), removeHs=False, sanitize=False)
            mols_3d = []   # (mol, name) already have 3D coords
            smiles_2d = []  # (smiles, name) need OpenBabel 3D generation

            for i, mol in enumerate(suppl):
                if mol is None:
                    continue
                mol_name = _extract_mol_name(mol, f'mol_{i}')
                if _has_3d_coords(mol):
                    mols_3d.append((mol, mol_name))
                else:
                    # Extract SMILES for OpenBabel 3D generation
                    try:
                        mol_san = Chem.RWMol(mol)
                        Chem.SanitizeMol(mol_san, catchErrors=True)
                        smiles = Chem.MolToSmiles(mol_san)
                        if smiles:
                            smiles_2d.append((smiles, mol_name))
                            logger.info(f"2D molecule '{mol_name}': queued for 3D generation")
                    except Exception as e:
                        logger.warning(f"Could not extract SMILES for '{mol_name}': {e}")

            logger.info(f"SDF upload: {len(mols_3d)} 3D molecules, {len(smiles_2d)} 2D molecules")

            if not mols_3d and not smiles_2d:
                raise HTTPException(400, "No valid molecules found in SDF file")

            prep_start = datetime.now()

            # Write 3D molecules, then adjust protonation at target pH via OpenBabel.
            # Strip all input SDF properties — GNINA echoes them back, and multi-line
            # property values (e.g. Maestro s_m_subgroup_title) corrupt the output SDF.
            # OpenBabel -p adjusts ionisation states (COO-, NH3+) without regenerating 3D.
            three_d_path = work_dir / "ligands_3d.sdf"
            three_d_raw_path = work_dir / "ligands_3d_raw.sdf"
            if mols_3d:
                writer_3d = Chem.SDWriter(str(three_d_raw_path))
                for mol, _ in mols_3d:
                    for prop in list(mol.GetPropsAsDict().keys()):
                        mol.ClearProp(prop)
                    writer_3d.write(mol)
                writer_3d.close()
                # Adjust protonation at target pH (preserves heavy-atom 3D coordinates)
                ph_proc = await asyncio.create_subprocess_exec(
                    'obabel', str(three_d_raw_path), '-O', str(three_d_path), '-p', str(ph),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, ph_stderr = await asyncio.wait_for(ph_proc.communicate(), timeout=120)
                if ph_proc.returncode != 0 or not three_d_path.exists():
                    logger.warning("OpenBabel pH adjustment for 3D ligands failed: %s — using raw input",
                                   ph_stderr.decode(errors='replace'))
                    three_d_path = three_d_raw_path
                else:
                    logger.info("OpenBabel pH %.1f adjustment applied to %d 3D molecules", ph, len(mols_3d))

            # Generate 3D for 2D molecules via existing OpenBabel pipeline
            two_d_prepared_path = work_dir / "ligands_from_2d.sdf"
            if smiles_2d:
                await job_processor.update_progress(
                    job_id,
                    status=JobStatus.PREPARING,
                    message=f"Generating 3D coordinates for {len(smiles_2d)} 2D molecules...",
                    current_stage="3D Generation"
                )
                success_2d, total_2d = await job_processor.prepare_ligands_from_smiles(
                    job_id=job_id,
                    smiles_with_ids=smiles_2d,
                    output_path=str(two_d_prepared_path),
                    ph=ph
                )
                logger.info(f"3D generation: {success_2d}/{total_2d} 2D molecules succeeded")

            active_jobs[job_id].timings['preparation'] = (datetime.now() - prep_start).total_seconds()

            # Combine 3D + prepared-2D into final ligand file
            with open(str(ligand_path), 'w') as out_f:
                if mols_3d and three_d_path.exists():
                    out_f.write(three_d_path.read_text())
                if smiles_2d and two_d_prepared_path.exists():
                    out_f.write(two_d_prepared_path.read_text())

            # Validate combined file
            verify_suppl = Chem.SDMolSupplier(str(ligand_path))
            mol_count = sum(1 for mol in verify_suppl if mol is not None)

            if mol_count == 0:
                raise HTTPException(400, "No valid 3D molecules could be prepared from SDF file")

            active_jobs[job_id].total_ligands = mol_count
            logger.info(f"SDF preparation complete: {mol_count} molecules ready for docking")
        
        # Run docking
        dock_start = datetime.now()
        docked_path = await job_processor.run_docking_job(
            job_id=job_id,
            receptor_path=str(receptor_path),
            reference_path=str(reference_path),
            ligand_path=str(ligand_path),
            output_dir=str(work_dir),
            num_poses=num_poses,
            exhaustiveness=exhaustiveness,
            cnn_scoring=cnn_scoring,
            seed=seed
        )
        active_jobs[job_id].timings['docking'] = (datetime.now() - dock_start).total_seconds()
        
        # Restore formal charges stripped by GNINA (e.g. COO⁻, NH₃⁺) via obabel -p.
        # GNINA writes geometrically correct poses but omits M CHG records; without them
        # DataWarrior adds implicit H to undervalenced atoms (COO → COOH).
        docked_ph_path = work_dir / "docked_ph.sdf"
        ph_proc = await asyncio.create_subprocess_exec(
            'obabel', str(docked_path), '-O', str(docked_ph_path), '-p', str(ph),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, ph_stderr = await asyncio.wait_for(ph_proc.communicate(), timeout=120)
        if ph_proc.returncode == 0 and docked_ph_path.exists():
            logger.info("Applied pH %.1f charge correction to GNINA output", ph)
            docked_path = str(docked_ph_path)
        else:
            logger.warning("obabel pH correction on GNINA output failed: %s",
                           ph_stderr.decode(errors='replace'))

        # Sort results
        await job_processor.update_progress(
            job_id,
            status=JobStatus.FINALIZING,
            progress=90,
            message="Sorting and formatting results...",
            current_stage="Finalization"
        )

        final_path = work_dir / "docking_results.sdf"
        num_poses_out = job_processor.sort_and_filter_results(
            input_path=docked_path,
            output_path=str(final_path),
            sort_by=sort_by,
        )

        if num_poses_out == 0:
            raise HTTPException(500, "Docking produced no poses. Check server logs for GPU errors.")

        # Optional MCS RMSD annotation
        if mcs_rmsd:
            await job_processor.update_progress(
                job_id, progress=92, message="Calculating MCS RMSD...",
                current_stage="MCS RMSD"
            )
            job_processor.add_mcs_rmsd(str(final_path), str(reference_path))

        # Optional shape similarity annotation
        if shape_sim:
            await job_processor.update_progress(
                job_id, progress=93, message="Calculating shape similarity...",
                current_stage="Shape Sim"
            )
            job_processor.add_shape_sim(str(final_path), str(reference_path))

        # Optional 2D Morgan ECFP4 similarity to reference
        if ref_sim:
            await job_processor.update_progress(
                job_id, progress=93, message="Calculating 2D similarity to reference...",
                current_stage="Ref Sim"
            )
            job_processor.add_ref_sim(str(final_path), str(reference_path))

        # Optional PoseBusters validation
        if posebusters:
            await job_processor.update_progress(
                job_id, progress=94, message="Running PoseBusters validation...",
                current_stage="PoseBusters"
            )
            job_processor.add_posebusters_flags(str(final_path))

        # Optional PyMOL session
        pse_path = None
        if generate_pymol:
            await job_processor.update_progress(
                job_id, progress=95, message="Generating PyMOL session...",
                current_stage="PyMOL"
            )
            pse_path = await job_processor.generate_pymol_session(
                work_dir=str(work_dir),
                receptor_path=str(receptor_path),
                reference_path=str(reference_path),
                docking_results_sdf=str(final_path),
                session_name=session_name,
            )

        total_time = (datetime.now() - start_time).total_seconds()
        active_jobs[job_id].timings['total'] = total_time

        await job_processor.update_progress(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            message=f"Completed! {num_poses_out} poses in {total_time:.1f}s"
                    + (" (+ PyMOL session)" if pse_path else ""),
            result_file=str(final_path)
        )
        logger.info(f"Job {job_id} completed: {num_poses_out} poses in {total_time:.1f}s")

        stem = session_name if session_name else f"docking_{job_id}"

        if pse_path:
            zip_path = work_dir / f"{stem}.zip"
            with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(str(final_path), f"{stem}.sdf")
                zf.write(pse_path, f"{stem}.pse")
            return FileResponse(
                path=str(zip_path),
                filename=f"{stem}.zip",
                media_type="application/zip",
            )

        return FileResponse(
            path=str(final_path),
            filename=f"{stem}.sdf",
            media_type="application/octet-stream",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        active_jobs[job_id].status = JobStatus.FAILED
        active_jobs[job_id].error = str(e)
        raise HTTPException(500, f"Docking failed: {str(e)}")
    
    finally:
        # Schedule cleanup (keep files for a bit for debugging)
        async def cleanup():
            await asyncio.sleep(300)  # Keep for 5 minutes
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            if job_id in active_jobs:
                del active_jobs[job_id]
        
        asyncio.create_task(cleanup())


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    
    if job_id not in job_websockets:
        job_websockets[job_id] = []
    job_websockets[job_id].append(websocket)
    
    try:
        while True:
            # Send current status if available
            if job_id in active_jobs:
                job = active_jobs[job_id]
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "message": job.message,
                    "total_ligands": job.total_ligands,
                    "processed_ligands": job.processed_ligands,
                    "current_stage": job.current_stage,
                    "timings": job.timings
                })
            
            # Wait for client ping or timeout
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if job_id in job_websockets and websocket in job_websockets[job_id]:
            job_websockets[job_id].remove(websocket)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a docking job."""
    if job_id not in active_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    
    job = active_jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "total_ligands": job.total_ligands,
        "processed_ligands": job.processed_ligands,
        "current_stage": job.current_stage,
        "error": job.error,
        "timings": job.timings
    }


if __name__ == "__main__":
    uvicorn.run(
        "gnina_app:app",
        host="130.237.250.75",
        port=9000,
        workers=1,  # Single worker for shared state
        reload=False,
        log_level="info"
    )