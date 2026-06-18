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
import base64
import os
import sys
import tempfile
import subprocess
import shutil
import json
import logging
import logging.handlers
import uuid
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
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdShapeHelpers, DataStructs
from rdkit import RDLogger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def _sanitize_mol(mol) -> None:
    """Sanitize an RDKit mol loaded with sanitize=False.

    Full SanitizeMol can fail on GNINA output (unusual atom types, charges).
    When it does, fall back to FastFindRings so that ring info is populated
    and Morgan fingerprints / MCS still work.
    """
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.FastFindRings(mol)

# Configure logging — WARNING+ to console/file (suppress verbose INFO)
_LOG_FILE = os.path.join(os.path.dirname(__file__), 'gnina_webapp.log')
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# Slim access log: one line per docking job, appended to a single file
_ACCESS_LOG_FILE = os.path.join(os.path.dirname(__file__), 'gnina_access.log')
access_logger = logging.getLogger('gnina.access')
access_logger.setLevel(logging.INFO)
access_logger.propagate = False
_access_handler = logging.FileHandler(_ACCESS_LOG_FILE, encoding='utf-8')
_access_handler.setFormatter(logging.Formatter('%(message)s'))
access_logger.addHandler(_access_handler)

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
                <div class="stat"><span>⚡</span><span>16 CPU Cores</span></div>
                <div class="stat"><span>🎮</span><span>1× RTX PRO 500 GPU</span></div>
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
                        <div class="gpu-name">GPU 0 • RTX PRO 500</div>
                        <div class="gpu-info">12 CPU cores assigned</div>
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
# CONFIGURATION
# ============================================================================
N_CPU = int(os.environ.get('N_CPU', os.cpu_count() or 1))
RESERVED_CPU = 4                   # Reserve for system/web server
WORKER_CPU = max(1, N_CPU - RESERVED_CPU)

def _detect_gpu_ids() -> List[int]:
    """Return list of CUDA device indices to use for docking.

    Priority:
    1. DOCK_GPUS env var  — comma-separated indices, e.g. "0,2"
    2. DOCK_GPU_ID env var — legacy single-index, e.g. "1"
    3. Auto-detect all CUDA devices via nvidia-smi
    4. Fall back to [0] if detection fails
    """
    env_gpus = os.environ.get('DOCK_GPUS', '').strip()
    if env_gpus:
        return [int(x) for x in env_gpus.split(',') if x.strip().isdigit()]
    env_single = os.environ.get('DOCK_GPU_ID', '').strip()
    if env_single.isdigit():
        return [int(env_single)]
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ids = [int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()]
            if ids:
                return ids
    except Exception:
        pass
    return [0]

DOCK_GPU_IDS: List[int] = _detect_gpu_ids()
N_GPU = len(DOCK_GPU_IDS)
CPU_PER_GPU = max(1, WORKER_CPU // N_GPU)

_NVIDIA_SMI = shutil.which(
    'nvidia-smi',
    path='/usr/lib/wsl/lib:/usr/bin:/usr/local/bin:' + os.environ.get('PATH', '')
) or 'nvidia-smi'

def _shorten_gpu_name(name: str) -> str:
    name = re.sub(r'^NVIDIA\s+', '', name)
    name = re.sub(r'\s+\w+ Generation\b.*', '', name)
    name = re.sub(r'\s+(Laptop|Desktop)\s+GPU$', '', name, flags=re.IGNORECASE)
    return name.strip()

def _detect_gpu_names() -> dict:
    try:
        result = subprocess.run(
            [_NVIDIA_SMI, '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            names = {}
            for line in result.stdout.splitlines():
                parts = line.split(',', 1)
                if len(parts) == 2:
                    names[int(parts[0].strip())] = _shorten_gpu_name(parts[1].strip())
            return names
    except Exception:
        pass
    return {}

GPU_NAMES: dict = _detect_gpu_names()

# Flexible docking: hard cap on side chains made movable per ligand. Keeps
# runtime and out_flex size bounded; gnina retains the closest residues.
FLEX_MAX_RESIDUES = int(os.environ.get('FLEX_MAX_RESIDUES', '8'))

# Working directories
WORK_DIR = Path("/tmp/gnina_work")
WORK_DIR.mkdir(exist_ok=True)

# GNINA binary path - update as needed
GNINA_PATH = os.environ.get('GNINA_PATH', '/opt/gnina/gnina')

# PyMOL binary for headless session generation (pymol -cq script.py)
PYMOL_PATH = os.environ.get('PYMOL_PATH', str(Path(sys.executable).parent / 'pymol'))

# Protein preparation (protprep.py) — runs in openmmdl conda env
def _find_openmmdl_python() -> str:
    """Locate the openmmdl conda env python, falling back to sys.executable."""
    import shutil, subprocess

    # 1. Try conda (via PATH or common install locations)
    conda_candidates = list(filter(None, [
        shutil.which("conda"),
        *[str(p) for p in [
            Path.home() / "Programs/miniconda3/bin/conda",
            Path.home() / "miniconda3/bin/conda",
            Path.home() / "anaconda3/bin/conda",
            Path("/opt/conda/bin/conda"),
        ] if Path(p).exists()]
    ]))
    for conda in conda_candidates:
        try:
            result = subprocess.run(
                [conda, "run", "-n", "openmmdl", "python", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and (exe := result.stdout.strip()):
                return exe
        except Exception:
            pass

    # 2. Fall back: look for the env's python directly on disk
    for candidate in [
        Path.home() / "Programs/miniconda3/envs/openmmdl/bin/python",
        Path.home() / "miniconda3/envs/openmmdl/bin/python",
        Path.home() / "anaconda3/envs/openmmdl/bin/python",
    ]:
        if candidate.exists():
            return str(candidate)

    return sys.executable

OPENMMDL_PYTHON = os.environ.get('OPENMMDL_PYTHON') or _find_openmmdl_python()
PROTPREP_SCRIPT = os.environ.get('PROTPREP_SCRIPT', str(Path(__file__).parent / 'protprep.py'))

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
    result_zip: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    cancelled: bool = False
    gnina_procs: list = field(default_factory=list)

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
    filename = filename.lstrip('.')
    if len(filename) > 200:
        filename = filename[:200]
    return filename or 'unnamed'


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


_RESIDUE_TOKEN_RE = re.compile(r'([A-Za-z])\s*[:/\- ]?\s*(\d+)')


def parse_residue_list(text: str) -> List[Tuple[str, int]]:
    """Parse a residue specification like "A123, A:125, B 45, C/678".

    Returns a list of (chain, resnum) tuples in input order, deduplicated.
    Raises ValueError on empty or unparseable input.
    """
    if not text or not text.strip():
        raise ValueError("residue list is empty")
    matches = _RESIDUE_TOKEN_RE.findall(text)
    if not matches:
        raise ValueError(
            "could not parse residues — expected e.g. 'A123, A:125, B 45'"
        )
    seen = set()
    out: List[Tuple[str, int]] = []
    for chain, num in matches:
        key = (chain.upper(), int(num))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def compute_residue_centroid(
    pdb_path: str,
    residues: List[Tuple[str, int]],
) -> Tuple[float, float, float]:
    """Compute the centroid of the specified residues from a PDB file.

    Uses Cα atom for each residue when present; otherwise falls back to the
    heavy-atom centroid of that residue (e.g. HETATM ligand residues).

    Raises ValueError if any requested residue is not found in the file.
    """
    wanted = {(c.upper(), n) for c, n in residues}
    found_ca: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    found_heavy: Dict[Tuple[str, int], List[Tuple[float, float, float]]] = {}

    with open(pdb_path, 'r', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            if len(line) < 54:
                continue
            atom_name = line[12:16].strip()
            chain = line[21:22].strip().upper()
            try:
                resnum = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (chain, resnum)
            if key not in wanted:
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else ''
            if not element:
                element = ''.join(ch for ch in atom_name if ch.isalpha())[:1].upper()
            if element == 'H':
                continue
            if atom_name == 'CA':
                found_ca[key] = (x, y, z)
            found_heavy.setdefault(key, []).append((x, y, z))

    missing = sorted(wanted - set(found_heavy.keys()))
    if missing:
        raise ValueError(
            "residue(s) not found in receptor: "
            + ', '.join(f"{c}{n}" for c, n in missing)
        )

    coords: List[Tuple[float, float, float]] = []
    for key in wanted:
        if key in found_ca:
            coords.append(found_ca[key])
        else:
            atoms = found_heavy[key]
            cx = sum(a[0] for a in atoms) / len(atoms)
            cy = sum(a[1] for a in atoms) / len(atoms)
            cz = sum(a[2] for a in atoms) / len(atoms)
            coords.append((cx, cy, cz))

    n = len(coords)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def _split_pdb_models(pdb_path: str) -> List[str]:
    """
    Split a (possibly multi-MODEL) PDB into one text block per MODEL.

    GNINA's --out_flex writes the moved flexible residues for each output pose,
    wrapped in MODEL/ENDMDL records in pose order. A single-pose run may omit the
    MODEL wrapper, in which case the whole file is treated as one block. Returned
    blocks carry only ATOM/HETATM/TER/CONECT lines (no MODEL/ENDMDL), ready to be
    re-wrapped with a fresh MODEL number.
    """
    try:
        with open(pdb_path, 'r') as f:
            text = f.read()
    except OSError:
        return []
    if not text.strip():
        return []

    keep_prefixes = ('ATOM', 'HETATM', 'TER', 'CONECT')
    models: List[str] = []
    current: List[str] = []
    in_model = False
    saw_model = False
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == 'MODEL':
            saw_model = True
            in_model = True
            current = []
        elif rec == 'ENDMDL':
            models.append('\n'.join(current))
            in_model = False
            current = []
        elif line.startswith(keep_prefixes):
            if in_model or not saw_model:
                current.append(line)
    # No MODEL wrapper at all → single implicit model from collected lines.
    if not saw_model and current:
        models.append('\n'.join(current))
    return [m for m in models if m.strip()]


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
    return ('\n\n$$$$\n'.join(merged) + '\n\n$$$$\n') if merged else ''


def parse_smiles_input(smiles_text: str) -> List[Tuple[str, str]]:
    """
    Parse SMILES input, supporting optional identifiers.

    Formats supported:
    - SMILES only: CCO
    - SMILES with space-separated ID: CCO ethanol
    - SMILES with comma-separated ID: CCO,ethanol
    - SMILES with tab-separated ID: CCO\tethanol

    Long SMILES that were hard-wrapped (e.g. pasted from an editor that breaks
    at 80 chars) are detected and rejoined: if a line is not a valid SMILES but
    concatenating it with the next line yields a valid SMILES, the two lines are
    merged and the second line's identifier (if any) is used.

    Returns: List of (smiles, identifier) tuples
    """
    # Normalize line endings
    smiles_text = smiles_text.replace('\r\n', '\n').replace('\r', '\n')
    lines = smiles_text.strip().split('\n')
    logger.info(f"parse_smiles_input: received {len(lines)} lines")

    # --- pass 1: parse each line individually, tracking whether the identifier
    #             was explicitly supplied or auto-generated ---
    raw: List[Tuple[str, str, bool]] = []  # (smiles, identifier, explicit_id)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        smiles = None
        identifier = None
        explicit_id = False

        if '\t' in line:
            parts = line.split('\t', 1)
            smiles = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                identifier = parts[1].strip()
                explicit_id = True
        elif ',' in line:
            comma_idx = line.rfind(',')
            after_comma = line[comma_idx + 1:].strip() if comma_idx >= 0 else ''
            if comma_idx > 0 and after_comma and re.match(r'^[\w][\w\-.]*$', after_comma):
                smiles = line[:comma_idx].strip()
                identifier = after_comma
                explicit_id = True
            else:
                smiles = line
        elif ' ' in line:
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                potential_id = parts[1].strip()
                if potential_id and re.match(r'^[\w][\w\-.]*$', potential_id):
                    smiles = parts[0].strip()
                    identifier = potential_id
                    explicit_id = True
                else:
                    smiles = line
            else:
                smiles = line
        else:
            smiles = line

        if not identifier:
            identifier = f"ligand_{i+1:04d}"
        identifier = re.sub(r'[^\w\-.]', '_', identifier)

        if smiles:
            raw.append((smiles, identifier, explicit_id))

    # --- pass 2: rejoin hard-wrapped SMILES ---
    # If line N is not a valid SMILES but N + (N+1) is, merge them and take
    # line N+1's identifier when it was explicitly supplied.
    results: List[Tuple[str, str]] = []
    i = 0
    while i < len(raw):
        smi, ident, explicit = raw[i]
        if i + 1 < len(raw) and Chem.MolFromSmiles(smi) is None:
            next_smi, next_ident, next_explicit = raw[i + 1]
            combined = smi + next_smi
            if Chem.MolFromSmiles(combined) is not None:
                final_ident = next_ident if next_explicit else ident
                results.append((combined, final_ident))
                logger.info(
                    f"  Merged wrapped SMILES lines {i}+{i+1}: "
                    f"ID='{final_ident}' SMILES='{combined[:60]}{'...' if len(combined) > 60 else ''}'")
                i += 2
                continue
        results.append((smi, ident))
        logger.info(f"  Parsed line {i}: ID='{ident}' SMILES='{smi[:60]}{'...' if len(smi) > 60 else ''}'")
        i += 1

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


def _rdkit_embed_and_minimize(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Embed a SMILES into 3D with ETKDGv3 and minimize with MMFF94s
    (falling back to UFF when MMFF parameters are missing).

    OpenBabel's --gen3d places bridging phosphorus atoms in square planar
    geometry (one bond at 180°) for triphosphates and similar — and MMFF94s
    steepest-descent can't climb out of it. ETKDG builds proper tetrahedral
    geometry for those cases.

    Returns: (molblock or None, error message or None).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"RDKit could not parse SMILES '{smiles}'"
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None, "ETKDG embedding failed"

        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props is not None else None
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is not None:
            ff.Minimize(maxIts=2000)

        return Chem.MolToMolBlock(mol) + '$$$$\n', None
    except Exception as e:
        return None, f"RDKit 3D generation error: {e}"


def prepare_single_ligand(args: Tuple[str, int, float, str]) -> Tuple[int, Optional[str], Optional[str], str]:
    """
    Prepare a single ligand from SMILES: protonate at pH with OpenBabel, then
    build 3D coordinates with RDKit ETKDGv3 and minimize with MMFF94s
    (falling back to OpenBabel --gen3d only if RDKit fails).

    Returns: (index, mol_block or None, error or None, identifier)

    This function must be at module level for multiprocessing to work.
    """
    smiles, idx, ph, identifier = args
    tmp_dir = None

    try:
        import subprocess
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix=f"lig_{idx}_")
        smi_file = os.path.join(tmp_dir, "input.smi")
        protonated_smi_file = os.path.join(tmp_dir, "protonated.smi")

        with open(smi_file, 'w') as f:
            f.write(f"{smiles} {identifier}\n")

        # Step 1: Protonate at pH with OpenBabel (-r strips salts / small fragments)
        cmd_protonate = [
            'obabel', smi_file,
            '-O', protonated_smi_file,
            '-r',
            '-p', str(ph),
        ]
        result1 = subprocess.run(cmd_protonate, capture_output=True, text=True, timeout=30)

        if not os.path.exists(protonated_smi_file) or os.path.getsize(protonated_smi_file) == 0:
            stderr_msg = (result1.stderr or 'No output').strip()
            return idx, None, f"Protonation failed for SMILES '{smiles}': {stderr_msg[:400]}", identifier

        with open(protonated_smi_file, 'r') as f:
            protonated_line = f.readline().strip()
        # SMI files use tab or whitespace between SMILES and name
        protonated_smiles = re.split(r'[\s\t]', protonated_line, 1)[0] if protonated_line else ''
        if not protonated_smiles:
            return idx, None, f"Protonation produced empty SMILES from '{smiles}'", identifier

        # Step 2: 3D embed + minimize with RDKit (primary)
        sdf_block, rdkit_err = _rdkit_embed_and_minimize(protonated_smiles)

        # Fallback: OpenBabel --gen3d best (no minimization) if RDKit failed
        if sdf_block is None:
            sdf_file = os.path.join(tmp_dir, "output.sdf")
            cmd_3d = ['obabel', protonated_smi_file, '-O', sdf_file, '--gen3d', 'best']
            result2 = subprocess.run(cmd_3d, capture_output=True, text=True, timeout=120)
            if os.path.exists(sdf_file) and os.path.getsize(sdf_file) > 50:
                with open(sdf_file, 'r') as f:
                    candidate = f.read()
                if candidate and '$$$$' in candidate and 'M  END' in candidate:
                    sdf_block = candidate
            if sdf_block is None:
                ob_err = (result2.stderr or 'unknown').strip()[:200]
                return idx, None, f"3D generation failed for '{smiles}': RDKit={rdkit_err}; OpenBabel={ob_err}", identifier

        # Normalize the title line to the identifier and strip properties
        lines = sdf_block.split('\n')
        if lines:
            lines[0] = identifier
        sdf_block = '\n'.join(lines)
        sdf_block = strip_sdf_properties(sdf_block)

        return idx, sdf_block, None, identifier

    except subprocess.TimeoutExpired:
        return idx, None, f"Timeout processing: {smiles[:50]}", identifier
    except Exception as e:
        return idx, None, f"Error processing {smiles[:50]}...: {str(e)}", identifier
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


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
        self.gpu_semaphores = {gpu_id: asyncio.Semaphore(1) for gpu_id in DOCK_GPU_IDS}
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
        reference_path: Optional[str] = None,
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
        gpu_id: int = 0,
        num_modes: int = 9,
        exhaustiveness: int = 8,
        cnn_scoring: str = 'rescore',
        autobox_add: float = 4.0,
        seed: int = 666,
        job_id: str = '',
        flexdist_ligand: Optional[str] = None,
        flexdist: Optional[float] = None,
        out_flex_path: Optional[str] = None,
        covalent_rec_atom: Optional[str] = None,
        covalent_lig_atom_pattern: Optional[str] = None,
        covalent_optimize_lig: bool = True,
    ) -> Tuple[bool, str]:
        """
        Run GNINA docking on a specific GPU.

        Binding site is defined either by reference_path (autobox) OR by
        center + size. Exactly one of those must be provided.

        Flexible docking: pass flexdist_ligand + flexdist to let side chains
        within `flexdist` Å of that ligand move during docking. The moved
        residue conformers (one per output pose, in lockstep with output_path)
        are written to out_flex_path as multi-MODEL PDB.

        Covalent docking: pass covalent_rec_atom (chain:resnum:atom_name) plus
        covalent_lig_atom_pattern (a SMARTS matching the ligand's reactive atom)
        to tether that ligand atom to the receptor atom during docking.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if reference_path and center:
            raise ValueError("dock_batch: pass either reference_path or center, not both")
        if not reference_path and not center:
            raise ValueError("dock_batch: must pass reference_path or center")

        cmd = [
            self.gnina_path,
            '-r', receptor_path,
            '-l', ligand_path,
            '-o', output_path,
        ]
        if reference_path:
            cmd += ['--autobox_ligand', reference_path,
                    '--autobox_add', str(autobox_add)]
        else:
            cx, cy, cz = center
            sx, sy, sz = size if size else (16.0, 16.0, 16.0)
            cmd += ['--center_x', f'{cx:.3f}',
                    '--center_y', f'{cy:.3f}',
                    '--center_z', f'{cz:.3f}',
                    '--size_x', f'{sx:.3f}',
                    '--size_y', f'{sy:.3f}',
                    '--size_z', f'{sz:.3f}']
        cmd += [
            '--num_modes', str(num_modes),
            '--exhaustiveness', str(exhaustiveness),
            '--cnn_scoring', cnn_scoring,
            '--cpu', str(CPU_PER_GPU),
            '--seed', str(seed)
        ]

        # Flexible docking: make near-ligand side chains movable and capture them.
        if flexdist_ligand and flexdist is not None:
            cmd += ['--flexdist_ligand', flexdist_ligand,
                    '--flexdist', str(flexdist)]
            # Cap flexible residues so runtime/output stay bounded for big pockets.
            cmd += ['--flex_max', str(FLEX_MAX_RESIDUES)]
            if out_flex_path:
                cmd += ['--out_flex', out_flex_path]

        # Covalent docking: tether the ligand atom matching the SMARTS pattern to
        # the named receptor atom (chain:resnum:atom_name).
        if covalent_rec_atom and covalent_lig_atom_pattern:
            cmd += ['--covalent_rec_atom', covalent_rec_atom,
                    '--covalent_lig_atom_pattern', covalent_lig_atom_pattern]
            if covalent_optimize_lig:
                cmd += ['--covalent_optimize_lig']

        # GNINA v1.3.2 Torch backend ignores --device; use CUDA_VISIBLE_DEVICES instead
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        async with self.gpu_semaphores[gpu_id]:
            logger.info(f"Starting docking on GPU {gpu_id}: {os.path.basename(ligand_path)}")
            logger.info(f"GNINA command: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            if job_id and job_id in active_jobs:
                active_jobs[job_id].gnina_procs.append(proc)
            stdout, stderr = await proc.communicate()
            if job_id and job_id in active_jobs and active_jobs[job_id].cancelled:
                return False, "cancelled"
            
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
        loop = asyncio.get_running_loop()
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

        if all_results:
            verify_suppl = Chem.SDMolSupplier(output_path)
            verified_count = sum(1 for mol in verify_suppl if mol is not None)
            logger.info(f"RDKit verification: reads {verified_count} molecules")
        else:
            logger.warning("No ligands prepared successfully; skipping RDKit verification of empty SDF")
        
        if errors:
            logger.warning(f"Ligand preparation errors: {errors}")
        
        return len(all_results), total

    async def generate_pymol_session(
        self,
        work_dir: str,
        receptor_path: str,
        docking_results_sdf: str,
        reference_path: Optional[str] = None,
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
        session_name: str = '',
        sort_by: str = 'minimizedAffinity',
        flex_blocks: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Generate a PyMOL session (.pse) from the docked SDF output.

        Each unique ligand becomes one multi-state PyMOL object (one state per pose),
        so the user can cycle through poses with PyMOL's state controls.
        For flexible docking, `flex_blocks` maps each pose's <FlexPoseID> to its
        moved side-chain PDB; a parallel multi-state `{obj}_flex` object is built so
        the flexed residues track the ligand's pose state.
        Pose files are written from the raw SDF text — never through SDWriter —
        to avoid bond-table corruption.

        Returns path to the .pse file, or None on failure.
        """
        PYMOL_EXE = PYMOL_PATH

        def _pymol_name(name: str) -> str:
            """Make a valid PyMOL object name."""
            s = re.sub(r'[^\w]', '_', name)
            if s and s[0].isdigit():
                s = 'm' + s
            return s or 'ligand'

        pymol_dir = os.path.join(work_dir, 'pymol_files')
        os.makedirs(pymol_dir, exist_ok=True)
        pse_filename = f'{session_name}.pse' if session_name else 'visualization.pse'
        pse_path = os.path.join(work_dir, pse_filename)

        # Read output SDF and group blocks by molecule name (first line).
        # Split on the '$$$$' delimiter *line* (consuming its trailing newline) so
        # each block starts exactly at its title line — which may legitimately be
        # empty for an unnamed ligand. Splitting on bare '$$$$' would leave that
        # delimiter newline on the block; a later strip() would then swallow an
        # empty title and mis-read the program line as the molecule name.
        with open(docking_results_sdf, 'r') as f:
            content = f.read()
        raw_blocks = [b for b in re.split(r'\$\$\$\$\r?\n?', content) if b.strip()]

        higher_is_better = sort_by in ('CNNscore', 'CNNaffinity', 'CNN_VS')

        def _block_score(block: str) -> float:
            for field in ([sort_by] if sort_by != 'minimizedAffinity' else []) + ['minimizedAffinity']:
                m = re.search(r'>\s*<' + re.escape(field) + r'>\s*\n\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', block)
                if m:
                    v = float(m.group(1))
                    return -v if higher_is_better else v
            return 0.0

        from collections import defaultdict
        ligand_blocks: Dict[str, list] = defaultdict(list)
        for block in raw_blocks:
            # First line of the block is the SDF title line (do NOT strip the
            # block first — that would drop an empty title and shift to line 2).
            mol_name = block.split('\n', 1)[0].strip() or 'unknown'
            ligand_blocks[mol_name].append(block.strip())

        # Write one SDF per ligand (all poses → multi-state object in PyMOL).
        # Sort each ligand's poses by minimizedAffinity (lower = better) so that
        # state 1 is always the best pose for that ligand.
        # For flexible docking, write a parallel multi-MODEL PDB of the moved side
        # chains in the same pose order so flex state N tracks ligand state N.
        ligand_entries = []  # (pymol_obj_name, sdf_path)
        flex_entries = []    # (pymol_obj_name, flex_pdb_path)
        for mol_name, blocks in ligand_blocks.items():
            blocks.sort(key=_block_score)
            obj_name = _pymol_name(mol_name)
            lig_sdf = os.path.join(pymol_dir, f'{obj_name}.sdf')
            with open(lig_sdf, 'w') as f:
                for b in blocks:
                    f.write(b.rstrip('\r\n ') + '\n\n$$$$\n')
            ligand_entries.append((obj_name, lig_sdf))

            if flex_blocks:
                flex_models = []
                for b in blocks:
                    pid = self._pose_flex_id(b)
                    if pid and flex_blocks.get(pid):
                        flex_models.append(flex_blocks[pid])
                if flex_models:
                    flex_pdb = os.path.join(pymol_dir, f'{obj_name}_flex.pdb')
                    with open(flex_pdb, 'w') as f:
                        for n, fm in enumerate(flex_models, start=1):
                            f.write(f"MODEL     {n:>4}\n{fm.rstrip(chr(10))}\nENDMDL\n")
                    flex_entries.append((obj_name, flex_pdb))

        # Build PyMOL Python script
        load_cmds = '\n'.join(
            f"cmd.load(r'{sdf}', '{obj}')\n"
            f"cmd.show('sticks', '{obj}')\n"
            f"cmd.hide('nonbonded', '{obj}')"
            for obj, sdf in ligand_entries
        )
        ligand_obj_list = ', '.join(f"'{obj}'" for obj, _ in ligand_entries)

        # Flexible-docking side chains: orange-carbon sticks, multi-state (tracks pose).
        if flex_entries:
            flex_load_cmds = "# --- Flexed receptor side chains (orange carbons) ---\n" + '\n'.join(
                f"cmd.load(r'{pdb}', '{obj}_flex')\n"
                f"cmd.show('sticks', '{obj}_flex')\n"
                f"cmd.hide('nonbonded', '{obj}_flex')\n"
                f"cmd.color('orange', '{obj}_flex and elem C')\n"
                f"cmd.color('atomic', '{obj}_flex and not elem C')\n"
                f"cmd.hide('sticks', '{obj}_flex and hydro and (neighbor (elem C))')\n"
                f"cmd.set('all_states', 0, '{obj}_flex')"
                for obj, pdb in flex_entries
            )
        else:
            flex_load_cmds = "# (no flexible side chains)"

        if reference_path:
            site_block = (
                f"# --- Reference ligand: green carbons, element colours, polar H shown ---\n"
                f"cmd.load(r'{reference_path}', 'reference_ligand')\n"
                f"cmd.show('sticks', 'reference_ligand')\n"
                f"cmd.hide('nonbonded', 'reference_ligand')\n"
                f"cmd.util.cbag('reference_ligand')\n"
                f"cmd.hide('sticks', 'reference_ligand and hydro and (neighbor (elem C))')"
            )
        elif center and size:
            cx, cy, cz = center
            sx, sy, sz = size
            site_block = (
                f"# --- Search box wireframe ---\n"
                f"cmd.pseudoatom('box_center', pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}])\n"
                f"cmd.hide('everything', 'box_center')\n"
                f"from pymol.cgo import LINEWIDTH, BEGIN, LINES, VERTEX, END, COLOR\n"
                f"_hx, _hy, _hz = {sx/2:.3f}, {sy/2:.3f}, {sz/2:.3f}\n"
                f"_cx, _cy, _cz = {cx:.3f}, {cy:.3f}, {cz:.3f}\n"
                f"_corners = [(_cx + dx*_hx, _cy + dy*_hy, _cz + dz*_hz)\n"
                f"             for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]\n"
                f"_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),\n"
                f"          (3,7),(4,5),(4,6),(5,7),(6,7)]\n"
                f"_obj = [LINEWIDTH, 2.0, COLOR, 1.0, 1.0, 0.0, BEGIN, LINES]\n"
                f"for a, b in _edges:\n"
                f"    _obj += [VERTEX, *_corners[a], VERTEX, *_corners[b]]\n"
                f"_obj += [END]\n"
                f"cmd.load_cgo(_obj, 'search_box')"
            )
        else:
            site_block = "# (no reference ligand or search box provided)"

        script = f"""from pymol import cmd

# --- Protein ---
cmd.load(r'{receptor_path}', 'protein')
cmd.hide('everything', 'protein')
cmd.show('cartoon', 'protein')
cmd.spectrum('count', 'rainbow', 'protein and name CA')

# --- Cofactors (non-polymer HETATM kept in receptor): magenta-carbon sticks ---
cmd.select('cofactors', 'protein and not polymer and not solvent')
if cmd.count_atoms('cofactors') > 0:
    cmd.show('sticks', 'cofactors')
    cmd.color('magenta', 'cofactors and elem C')
    cmd.color('atomic', 'cofactors and not elem C')
    cmd.hide('sticks', 'cofactors and hydro and (neighbor (elem C))')
cmd.delete('cofactors')

{site_block}

# --- Docked poses: polar H shown, non-polar H hidden ---
{load_cmds}

ligand_objs = [{ligand_obj_list}]
for obj in ligand_objs:
    cmd.hide('sticks', obj + ' and hydro and (neighbor (elem C))')

{flex_load_cmds}

# --- Binding-site residues within 5 Å of any docked pose: lines + labels ---
# Restrict to polymer atoms so cofactor sticks aren't overdrawn as lines/surface.
all_ligands_sel = ' or '.join(ligand_objs) if ligand_objs else 'none'
cmd.select('pocket_atoms', f'protein and polymer within 5 of ({{all_ligands_sel}})')
cmd.select('pocket_residues', f'byres pocket_atoms')
cmd.show('lines', 'pocket_residues')
cmd.hide('lines', 'pocket_residues and hydro and (neighbor (elem C))')
cmd.label('pocket_residues and name CA', '"%s%s" % (resn, resi)')

# Transparent light-grey surface on atoms within 5 Å only
cmd.show('surface', 'pocket_atoms')
cmd.set('surface_color', 'grey90', 'protein')
cmd.set('transparency', 0.5, 'protein')

cmd.deselect()
cmd.orient(all_ligands_sel if ligand_objs else 'protein')
cmd.zoom('pocket_residues', 5)
cmd.delete('pocket_atoms')
cmd.delete('pocket_residues')
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
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            return None
        except Exception as e:
            logger.error(f"PyMOL session generation failed: {e}")
            return None

    async def run_docking_job(
        self,
        job_id: str,
        receptor_path: str,
        ligand_path: str,
        output_dir: str,
        reference_path: Optional[str] = None,
        center: Optional[Tuple[float, float, float]] = None,
        size: Optional[Tuple[float, float, float]] = None,
        num_poses: int = 9,
        exhaustiveness: int = 8,
        cnn_scoring: str = 'rescore',
        seed: int = 666,
        flexdist_ligand: Optional[str] = None,
        flexdist: Optional[float] = None,
        covalent_rec_atom: Optional[str] = None,
        covalent_lig_atom_pattern: Optional[str] = None,
        covalent_optimize_lig: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """
        Run docking job with GPU load balancing.

        When flexdist_ligand + flexdist are given, side chains near that ligand
        flex during docking and their moved conformers are captured. Each pose
        in the merged SDF is tagged with a <FlexPoseID> field, and a sidecar
        JSON mapping that ID to the flex residue PDB block is written.

        When covalent_rec_atom + covalent_lig_atom_pattern are given, each ligand
        is covalently tethered to the named receptor atom during docking.

        Returns: (merged_results_sdf, flex_blocks_json | None)
        """
        flex_enabled = bool(flexdist_ligand and flexdist is not None)
        await self.update_progress(
            job_id,
            status=JobStatus.DOCKING,
            progress=35,
            message="Starting molecular docking...",
            current_stage="GPU Docking",
            processed_ligands=0,
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
        flex_files = []  # parallel to output_files; per-GPU out_flex PDBs (or None)

        for i in range(num_gpus_to_use):
            gpu_id = DOCK_GPU_IDS[i]
            start_idx = i * mols_per_gpu
            end_idx = min(start_idx + mols_per_gpu, total_mols)

            if start_idx >= total_mols:
                break

            batch_blocks = mol_blocks[start_idx:end_idx]

            # Write batch to file (text-based, preserving original format)
            batch_path = os.path.join(output_dir, f"ligands_gpu{gpu_id}.sdf")
            output_path = os.path.join(output_dir, f"docked_gpu{gpu_id}.sdf")
            flex_path = os.path.join(output_dir, f"flex_gpu{gpu_id}.pdb") if flex_enabled else None

            with open(batch_path, 'w') as f:
                f.write(''.join(batch_blocks))

            logger.info(f"GPU {gpu_id}: {len(batch_blocks)} ligands written to {batch_path}")

            # Create docking task
            task = self.engine.dock_batch(
                receptor_path=receptor_path,
                ligand_path=batch_path,
                output_path=output_path,
                reference_path=reference_path,
                center=center,
                size=size,
                gpu_id=gpu_id,
                num_modes=num_poses,
                exhaustiveness=exhaustiveness,
                cnn_scoring=cnn_scoring,
                seed=seed,
                job_id=job_id,
                flexdist_ligand=flexdist_ligand,
                flexdist=flexdist,
                out_flex_path=flex_path,
                covalent_rec_atom=covalent_rec_atom,
                covalent_lig_atom_pattern=covalent_lig_atom_pattern,
                covalent_optimize_lig=covalent_optimize_lig,
            )
            gpu_tasks.append(task)
            output_files.append(output_path)
            flex_files.append(flex_path)
        
        await self.update_progress(
            job_id,
            message=f"Docking {total_mols} ligands on {len(gpu_tasks)} GPU(s)..."
        )

        # Poll the per-GPU SDF outputs while docking runs so the UI shows
        # genuine per-ligand progress (and ETA can compute). gnina writes
        # num_poses pose-blocks per ligand, each terminated by "$$$$".
        async def _poll_docking_progress():
            try:
                while True:
                    await asyncio.sleep(2.0)
                    pose_total = 0
                    for path in output_files:
                        try:
                            with open(path, 'r') as f:
                                pose_total += f.read().count('$$$$')
                        except FileNotFoundError:
                            continue
                        except OSError:
                            continue
                    ligands_done = min(pose_total // max(num_poses, 1), total_mols)
                    prog = 35 + 50 * (ligands_done / total_mols) if total_mols else 35
                    await self.update_progress(
                        job_id,
                        progress=prog,
                        processed_ligands=ligands_done,
                        message=f"Docked {ligands_done}/{total_mols} ligands on {len(gpu_tasks)} GPU(s)..."
                    )
            except asyncio.CancelledError:
                return

        poll_task = asyncio.create_task(_poll_docking_progress())
        try:
            results = await asyncio.gather(*gpu_tasks, return_exceptions=True)
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except (asyncio.CancelledError, Exception):
                pass
        
        # Check for errors
        for i, result in enumerate(results):
            physical_gpu = DOCK_GPU_IDS[i]
            if isinstance(result, Exception):
                logger.error(f"GPU {physical_gpu} task failed: {result}")
            elif not result[0]:
                logger.error(f"GPU {physical_gpu} docking failed: {result[1]}")
        
        await self.update_progress(
            job_id,
            progress=85,
            message="Merging results..."
        )
        
        # Merge output files. For flexible docking, tag each pose with a unique
        # <FlexPoseID> and collect the matching out_flex residue block so the two
        # can be re-associated after pH correction and score sorting reorder poses.
        merged_path = os.path.join(output_dir, "docked_merged.sdf")
        flex_by_id: Dict[str, str] = {}
        total_poses = 0
        with open(merged_path, 'w') as outfile:
            for out_path, flex_path in zip(output_files, flex_files):
                if not os.path.exists(out_path):
                    logger.warning(f"Output file not found: {out_path}")
                    continue
                with open(out_path, 'r') as infile:
                    content = infile.read()

                if not flex_enabled:
                    poses_in_file = content.count('$$$$')
                    logger.info(f"Merging {out_path}: {poses_in_file} poses")
                    total_poses += poses_in_file
                    outfile.write(content)
                    continue

                # Merge GNINA's split structure/property blocks first so each pose
                # is a single $$$$ block, then associate one flex model per pose.
                content = _fix_split_sdf_blocks(content)
                pose_blocks = [b for b in content.split('$$$$') if b.strip()]
                gpu_tag = os.path.basename(out_path).replace('docked_', '').replace('.sdf', '')
                flex_models = _split_pdb_models(flex_path) if flex_path else []
                if flex_models and len(flex_models) != len(pose_blocks):
                    logger.warning(
                        f"Flex/pose count mismatch for {gpu_tag}: "
                        f"{len(flex_models)} flex models vs {len(pose_blocks)} poses — "
                        f"associating by position, extras dropped"
                    )
                for idx, block in enumerate(pose_blocks):
                    pose_id = f"{gpu_tag}_{idx}"
                    base = block.strip().rstrip('\r\n ')
                    outfile.write(base + f'\n\n> <FlexPoseID>\n{pose_id}\n\n$$$$\n')
                    if idx < len(flex_models):
                        flex_by_id[pose_id] = flex_models[idx]
                    total_poses += 1
                logger.info(f"Merging {out_path}: {len(pose_blocks)} poses, "
                            f"{len(flex_models)} flex models")

        logger.info(f"Merged total: {total_poses} poses")

        flex_map_path = None
        if flex_enabled and flex_by_id:
            flex_map_path = os.path.join(output_dir, "flex_blocks.json")
            with open(flex_map_path, 'w') as f:
                json.dump(flex_by_id, f)
            logger.info(f"Captured {len(flex_by_id)} flex residue blocks → {flex_map_path}")

        return merged_path, flex_map_path
    
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

        # Parse each raw block directly with MolFromMolBlock to guarantee we process
        # exactly as many molecules as there are raw_blocks — SDMolSupplier.__len__() can
        # miscount when GNINA output has M  CHG records or atypical V2000 formatting.
        logger.info(f"sort_and_filter_results: parsing {len(raw_blocks)} raw blocks")
        poses = []
        skipped_none = 0
        skipped_zero_atom = 0
        for i, raw_block in enumerate(raw_blocks):
            mol = Chem.MolFromMolBlock(raw_block.strip(), removeHs=False, sanitize=False)
            if mol is None:
                skipped_none += 1
                raw_name = raw_block.strip().split('\n')[0]
                logger.warning(f"sort_and_filter_results: mol[{i}] ('{raw_name}') returned None — skipped")
                continue
            if mol.GetNumAtoms() == 0:
                skipped_zero_atom += 1
                raw_name = raw_block.strip().split('\n')[0]
                logger.warning(f"sort_and_filter_results: mol[{i}] ('{raw_name}') has 0 atoms — skipped")
                continue

            mol_name = _extract_mol_name(mol, f"mol_{i}")
            raw_block = raw_block  # already have it

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
        if sort_by not in ('minimizedAffinity', 'minimizedRMSD') and docked_count == 0:
            logger.warning(
                "sort_by=%r but no poses have that field — all scores are 0.0 and sort order is undefined. "
                "GNINA does not write CNN_VS; use minimizedAffinity or CNNscore instead.", sort_by
            )

        if max_poses:
            poses = poses[:max_poses]

        # Write combined SDF from raw GNINA blocks (preserves exact format).
        # DockingRank is per-ligand (1 = best pose of that ligand) so that it
        # matches the PyMOL state number, which PoseViewer uses for navigation.
        from collections import defaultdict as _dd
        ligand_rank: Dict[str, int] = _dd(int)
        with open(output_path, 'w') as combined_f:
            for score, mol, mol_name, has_score, raw_block in poses:
                ligand_rank[mol_name] += 1
                pose_rank = ligand_rank[mol_name]

                if raw_block is not None:
                    # strip() removes both leading \n (from \n$$$$\n join) and trailing whitespace.
                    # Leading \n would shift the V2000 header lines and corrupt the structure.
                    base = raw_block.strip()
                    # Add Structure_ID if not already present in the GNINA output block.
                    # Use \n\n so there is a blank line before the field tag (SDF spec).
                    if '> <Structure_ID>' not in base and '> <Structure ID>' not in base:
                        base += f'\n\n> <Structure_ID>\n{mol_name}\n'
                    # Re-strip so DockingRank always gets a proper blank line before it.
                    block_text = base.rstrip('\r\n ') + f'\n\n> <DockingRank>\n{pose_rank}\n\n$$$$\n'
                else:
                    # Fallback through RDKit (shouldn't happen for GNINA output)
                    mol.SetIntProp('DockingRank', pose_rank)
                    block_text = Chem.MolToMolBlock(mol)
                    for pname in mol.GetPropNames():
                        block_text += f'> <{pname}>\n{mol.GetProp(pname)}\n\n'
                    block_text += f'> <Structure_ID>\n{mol_name}\n\n$$$$\n'

                combined_f.write(block_text)

        logger.info(f"Wrote {len(poses)} poses to {output_path}")
        return len(poses)

    @staticmethod
    def _pose_flex_id(block: str) -> Optional[str]:
        """Extract the <FlexPoseID> data field from an SDF pose block, if present."""
        m = re.search(r'>\s*<FlexPoseID>\s*\n\s*(\S+)', block)
        return m.group(1) if m else None

    def build_flex_pdb(self, final_sdf: str, flex_blocks: Dict[str, str],
                       output_path: str) -> int:
        """
        Write the flexed side chains as a multi-MODEL PDB aligned with the final,
        score-sorted poses: MODEL N holds the moved residues for pose N in
        `final_sdf`. Each pose carries a <FlexPoseID> field linking it to a block
        in `flex_blocks`. Returns the number of MODELs written.
        """
        with open(final_sdf) as f:
            blocks = [b for b in f.read().split('$$$$') if b.strip()]

        models_written = 0
        with open(output_path, 'w') as out:
            out.write("REMARK  Flexible receptor side chains from GNINA flexible docking\n")
            out.write("REMARK  MODEL n corresponds to pose n in the results SDF\n")
            for i, block in enumerate(blocks, start=1):
                pose_id = self._pose_flex_id(block)
                flex = flex_blocks.get(pose_id) if pose_id else None
                if not flex:
                    continue
                lig_name = block.strip().split('\n')[0].strip() or 'pose'
                out.write(f"MODEL     {i:>4}\n")
                out.write(f"REMARK    ligand={lig_name} FlexPoseID={pose_id}\n")
                out.write(flex.rstrip('\n') + '\n')
                out.write("ENDMDL\n")
                models_written += 1
        return models_written

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
        _sanitize_mol(ref_mol)

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
                _sanitize_mol(pose_mol)

                mcs_result = rdFMCS.FindMCS(
                    [ref_mol, pose_mol],
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareOrder,
                    ringMatchesRingOnly=True,
                    completeRingsOnly=False,
                    timeout=2,
                )

                if mcs_result.numAtoms < 3:
                    rmsd_str = 'N/A'
                else:
                    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                    # Cap matches to avoid O(ref²×pose²) explosion on symmetric scaffolds.
                    ref_matches = ref_mol.GetSubstructMatches(mcs_mol, uniquify=False, maxMatches=16)
                    pose_matches = pose_mol.GetSubstructMatches(mcs_mol, uniquify=False, maxMatches=16)

                    if not ref_matches or not pose_matches:
                        rmsd_str = 'N/A'
                    else:
                        ref_conf = ref_mol.GetConformer()
                        pose_conf = pose_mol.GetConformer()
                        best_rmsd = float('inf')
                        for ref_match in ref_matches:
                            ref_coords = np.array([list(ref_conf.GetAtomPosition(i)) for i in ref_match])
                            for pose_match in pose_matches:
                                pose_coords = np.array([list(pose_conf.GetAtomPosition(i)) for i in pose_match])
                                rmsd = float(np.sqrt(((ref_coords - pose_coords) ** 2).sum(axis=1).mean()))
                                if rmsd < best_rmsd:
                                    best_rmsd = rmsd
                        rmsd_str = f'{best_rmsd:.4f}'
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
        _sanitize_mol(ref_mol)
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
                _sanitize_mol(pose_mol)

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
        _sanitize_mol(ref_mol)
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
                _sanitize_mol(pose_mol)
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

    def add_posebusters_flags(self, sdf_path: str, receptor_path: str) -> Tuple[int, str]:
        """
        Annotates each pose in sdf_path with PB_Flags — the count of PoseBusters
        checks that failed (0 = all pass). Runs with config='dock' which covers
        internal geometry (bond lengths, angles, planarity, clashes within the
        ligand) AND intermolecular checks against the receptor (protein-ligand
        distance, volume overlap with protein/cofactors).
        Modifies sdf_path in-place.
        Returns (number of poses evaluated, failure-summary string for the run log).
        """
        try:
            from posebusters import PoseBusters
        except ImportError:
            logger.error("add_posebusters_flags: posebusters is not installed")
            return 0, ""

        with open(sdf_path, 'r') as f:
            content = f.read()
        blocks = [b for b in content.split('$$$$') if b.strip()]

        try:
            pb = PoseBusters(config='dock')
            df = pb.bust(sdf_path, mol_cond=receptor_path, full_report=False)
        except Exception as e:
            logger.error("add_posebusters_flags: PoseBusters failed: %s", e)
            return 0, ""

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

        # Build failure summary for run log and server log
        pb_summary = ""
        if not df.empty:
            fail_totals = (df[bool_cols] == False).sum().sort_values(ascending=False)  # noqa: E712
            failing = fail_totals[fail_totals > 0]
            if not failing.empty:
                pb_summary = failing.to_string()
                logger.warning("add_posebusters_flags: most common failures across all poses:\n%s", pb_summary)

        logger.warning("add_posebusters_flags: evaluated %d/%d poses in %s",
                       annotated, len(blocks), sdf_path)
        return annotated, pb_summary

    async def add_plif_sim(self, sdf_path: str, receptor_path: str, reference_path: str) -> int:
        """
        Annotates each pose in sdf_path with PLIF_Sim — ProLIF InteractionFingerprint
        Tanimoto similarity to the reference ligand.
        Runs in a thread pool so the event loop stays responsive. Modifies sdf_path
        in-place. Returns number of poses annotated.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._add_plif_sim_sync, sdf_path, receptor_path, reference_path
        )

    def _add_plif_sim_sync(self, sdf_path: str, receptor_path: str, reference_path: str) -> int:
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import MDAnalysis as mda
            import prolif
        except ImportError as e:
            logger.error("add_plif_sim: prolif not installed: %s", e)
            return 0

        # Load protein — retry with NoImplicit=False if RDKit valence check fails
        # (some PDB files have atoms with unusual connectivity that RDKit rejects
        # when NoImplicit=True enforces strict valence on explicit-H atoms).
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = mda.Universe(receptor_path)
                prot_ag = u.select_atoms("protein")
                try:
                    protein = prolif.Molecule.from_mda(prot_ag)
                except Exception:
                    protein = prolif.Molecule.from_mda(prot_ag, NoImplicit=False)
        except Exception as e:
            logger.error("add_plif_sim: failed to load receptor: %s", e)
            return 0

        # Load reference ligand
        ref_supp = Chem.SDMolSupplier(reference_path, removeHs=False, sanitize=False)
        ref_rdmol = next((m for m in ref_supp if m is not None and m.GetNumAtoms() > 0), None)
        if ref_rdmol is None:
            logger.error("add_plif_sim: could not load reference ligand from %s", reference_path)
            return 0

        try:
            ref_lig = prolif.Molecule.from_rdkit(Chem.AddHs(ref_rdmol, addCoords=True))
        except Exception as e:
            logger.error("add_plif_sim: failed to prepare reference ligand: %s", e)
            return 0

        with open(sdf_path, 'r') as f:
            content = f.read()
        blocks = [b for b in content.split('$$$$') if b.strip()]

        # Build list of valid ProLIF ligands, tracking which block index each came from
        pose_ligs = []
        valid_indices = []
        for i, block in enumerate(blocks):
            try:
                mol = Chem.MolFromMolBlock(block.strip(), removeHs=False, sanitize=False)
                if mol is not None and mol.GetNumAtoms() > 0:
                    pose_ligs.append(prolif.Molecule.from_rdkit(mol))
                    valid_indices.append(i)
            except Exception:
                pass

        # Run reference + all poses in a single Fingerprint instance so that
        # all bitvectors share the same residue-interaction column set and are
        # therefore the same length (required by TanimotoSimilarity).
        sims: Dict[int, float] = {}
        if pose_ligs:
            try:
                fp = prolif.Fingerprint()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fp.run_from_iterable([ref_lig] + pose_ligs, protein, progress=False)
                all_bvs = fp.to_bitvectors()
                ref_bv = all_bvs[0]
                for idx, bv in zip(valid_indices, all_bvs[1:]):
                    sims[idx] = DataStructs.TanimotoSimilarity(ref_bv, bv)
            except Exception as e:
                logger.error("add_plif_sim: fingerprint batch failed: %s", e)

        annotated = 0
        new_blocks = []
        for i, block in enumerate(blocks):
            sim = sims.get(i)
            sim_str = f'{sim:.4f}' if sim is not None else 'N/A'
            if sim is not None:
                annotated += 1
            new_blocks.append(block.strip() + f'\n\n> <PLIF_Sim>\n{sim_str}\n')

        with open(sdf_path, 'w') as f:
            for b in new_blocks:
                f.write(b.rstrip('\r\n ') + '\n\n$$$$\n')

        logger.info("add_plif_sim: annotated %d/%d poses in %s", annotated, len(blocks), sdf_path)
        return annotated




# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Clean up any work directories left over from a previous run
    if WORK_DIR.exists():
        for orphan in WORK_DIR.iterdir():
            shutil.rmtree(orphan, ignore_errors=True)
    WORK_DIR.mkdir(exist_ok=True)
    logger.warning(f"Starting GNINA Docking Server")
    yield
    logger.warning("Shutting down...")


app = FastAPI(
    title="GNINA Molecular Docking",
    description="High-performance molecular docking with GPU acceleration",
    version="2.0.0",
    lifespan=lifespan
)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

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


@app.get("/system-info")
async def system_info():
    """Return hardware info for the UI stats bar."""
    gpu_list = [{"id": gid, "name": GPU_NAMES.get(gid, f"GPU {gid}")} for gid in DOCK_GPU_IDS]
    return {"cpus": N_CPU, "gpus": gpu_list}


@app.post("/dock")
async def dock_molecules(
    request: Request,
    receptor: UploadFile = File(..., description="Protein structure in PDB format"),
    reference: Optional[UploadFile] = File(None, description="Reference ligand for binding site (SDF)"),
    site_residues: Optional[str] = Form(None, description="Residue list 'A123, B45' (alternative to reference)"),
    site_x: Optional[float] = Form(None, description="Search-box centre X (Å); pair with site_y/site_z to bypass reference/residues"),
    site_y: Optional[float] = Form(None, description="Search-box centre Y (Å)"),
    site_z: Optional[float] = Form(None, description="Search-box centre Z (Å)"),
    box_size: float = Form(16.0, ge=5, le=60, description="Cubic search box edge (Å) for residue/xyz mode"),
    ligand_file: Optional[UploadFile] = File(None, description="Ligands to dock (SDF)"),
    ligand_smiles: Optional[str] = Form(None, description="SMILES strings, one per line"),
    skip_ligprep: bool = Form(False, description="Skip ligand preparation: use uploaded SDF as-is (no pH adjustment, no 3D generation)"),
    ph: float = Form(7.4, ge=0, le=14, description="pH for protonation"),
    num_poses: int = Form(9, ge=1, le=50, description="Number of poses per ligand"),
    exhaustiveness: int = Form(8, ge=1, le=64, description="Search exhaustiveness"),
    cnn_scoring: Literal['rescore', 'refine', 'score_only', 'none'] = Form('rescore'),
    seed: int = Form(666),
    flexible: bool = Form(False, description="Flexible docking: flex side chains near the reference ligand"),
    flexdist: float = Form(3.5, ge=0, le=10, description="Side chains within this many Å of the reference ligand flex"),
    covalent: bool = Form(False, description="Covalent docking: tether the ligand reactive atom to a receptor residue atom"),
    covalent_chain: Optional[str] = Form(None, description="Chain ID of the reacting receptor residue (e.g. A)"),
    covalent_resnum: Optional[str] = Form(None, description="Residue number of the reacting receptor residue"),
    covalent_atom: str = Form('SG', description="Atom name of the reacting receptor residue (e.g. SG for Cys)"),
    covalent_smarts: Optional[str] = Form(None, description="SMARTS matching the ligand reactive atom (e.g. [C,c]=O)"),
    covalent_optimize: bool = Form(True, description="UFF-optimize the covalent ligand+residue complex"),
    sort_by: Literal['minimizedAffinity', 'CNNscore', 'CNNaffinity', 'CNN_VS'] = Form('minimizedAffinity'),
    generate_pymol: bool = Form(False),
    mcs_rmsd: bool = Form(False),
    shape_sim: bool = Form(False),
    ref_sim: bool = Form(False),
    posebusters: bool = Form(False),
    plif_sim: bool = Form(False),
    session_name: str = Form(''),
    client_job_id: Optional[str] = Form(None, description="Client-generated job ID for WebSocket tracking"),
):
    """
    Perform molecular docking with GNINA.

    Accepts either an SDF file with ligands or a list of SMILES strings.
    Binding site is defined either by a reference ligand SDF (autobox) or
    by a list of residues whose centroid defines the search-box centre.
    """
    # Sanitize session name for use in filenames
    session_name = re.sub(r'[^\w\-]', '_', session_name.strip()).strip('_')

    # Validate inputs
    if not receptor.filename:
        raise HTTPException(400, "Receptor PDB file is required")

    has_reference = bool(reference is not None and reference.filename)
    has_residues = bool(site_residues is not None and site_residues.strip())
    xyz_provided = [v for v in (site_x, site_y, site_z) if v is not None]
    if xyz_provided and len(xyz_provided) != 3:
        raise HTTPException(400, "Provide all three of site_x, site_y, site_z")
    has_xyz = len(xyz_provided) == 3

    n_modes = sum([has_reference, has_residues, has_xyz])
    if n_modes == 0:
        raise HTTPException(
            400,
            "Provide a binding-site source: reference ligand SDF, residue list, or xyz coordinates"
        )
    if n_modes > 1:
        raise HTTPException(
            400,
            "Provide only one binding-site source: reference ligand, residue list, or xyz coordinates"
        )

    # Flexible docking reuses the reference ligand as the flexdist ligand, so it
    # is only available in reference-ligand mode.
    if flexible and not has_reference:
        raise HTTPException(
            400,
            "Flexible docking requires the reference-ligand binding-site mode "
            "(the reference ligand defines which side chains flex)."
        )

    # Covalent docking: build the receptor atom spec (chain:resnum:atom_name)
    # and validate the SMARTS pattern for the ligand reactive atom.
    covalent_rec_atom: Optional[str] = None
    if covalent:
        resnum = (covalent_resnum or '').strip()
        chain = (covalent_chain or '').strip()
        atom = (covalent_atom or '').strip() or 'SG'
        smarts = (covalent_smarts or '').strip()
        if not resnum:
            raise HTTPException(400, "Covalent docking requires the reacting residue number")
        if not smarts:
            raise HTTPException(
                400,
                "Covalent docking requires a SMARTS pattern for the ligand reactive atom"
            )
        if Chem.MolFromSmarts(smarts) is None:
            raise HTTPException(400, f"Invalid covalent SMARTS pattern: {smarts!r}")
        # gnina expects chain:resnum:atom_name; the chain may be blank for
        # single-chain receptors.
        covalent_rec_atom = f"{chain}:{resnum}:{atom}"

    has_file = ligand_file is not None and ligand_file.filename
    has_smiles = ligand_smiles is not None and ligand_smiles.strip()
    
    if not has_file and not has_smiles:
        raise HTTPException(400, "Provide either ligand_file (SDF) or ligand_smiles")
    
    # Create job — use client-supplied ID if valid and not already active, otherwise generate one
    if client_job_id and re.match(r'^[a-zA-Z0-9_-]{4,32}$', client_job_id) and client_job_id not in active_jobs:
        job_id = client_job_id
    else:
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

        # Resolve binding-site definition
        reference_path: Optional[Path] = None
        site_center: Optional[Tuple[float, float, float]] = None
        site_size: Optional[Tuple[float, float, float]] = None

        if has_reference:
            reference_path = work_dir / secure_filename(reference.filename)
            content = await reference.read()
            reference_path.write_bytes(content)
        elif has_xyz:
            site_center = (float(site_x), float(site_y), float(site_z))
            site_size = (box_size, box_size, box_size)
            logger.info(
                f"Job {job_id}: xyz-defined site centre={site_center}, size={site_size}"
            )
        else:
            try:
                residues = parse_residue_list(site_residues)
                site_center = compute_residue_centroid(str(receptor_path), residues)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            site_size = (box_size, box_size, box_size)
            logger.info(
                f"Job {job_id}: residue-defined site centre={site_center}, "
                f"size={site_size}, residues={residues}"
            )

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
            active_jobs[job_id].total_ligands = success_count

            if success_count == 0:
                # Log all errors for debugging
                logger.error(f"All ligand preparations failed.")
                raise HTTPException(400, f"Failed to prepare any ligands from SMILES.")
        
        else:
            # Save uploaded SDF to raw path
            content = await ligand_file.read()
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = content.decode('latin-1')

            raw_sdf_path = work_dir / "ligands_uploaded_raw.sdf"
            raw_sdf_path.write_text(content_str)

            prep_start = datetime.now()

            if skip_ligprep:
                # Filter to blocks that have real 3D coordinates and strip all SDF
                # properties. This handles Schrodinger/Maestro SDFs that use split
                # blocks (structure + properties as separate $$$$ entries) — gnina
                # would try to dock each block independently without this step.
                # Coordinates and protonation are preserved unchanged.
                suppl_skip = Chem.SDMolSupplier(str(raw_sdf_path), removeHs=False, sanitize=False)
                kept = 0
                with Chem.SDWriter(str(ligand_path)) as _w:
                    for _i, _mol in enumerate(suppl_skip):
                        if _mol is None or not _has_3d_coords(_mol):
                            continue
                        _name = _extract_mol_name(_mol, f'mol_{_i}')
                        for _p in list(_mol.GetPropsAsDict().keys()):
                            _mol.ClearProp(_p)
                        _mol.SetProp('_Name', _name)
                        _w.write(_mol)
                        kept += 1
                logger.info("skip_ligprep=True: kept %d 3D molecules from uploaded SDF", kept)
            else:
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

                # Write 3D molecules, then adjust protonation at target pH via OpenBabel.
                # Strip all input SDF properties — GNINA echoes them back, and multi-line
                # property values (e.g. Maestro s_m_subgroup_title) corrupt the output SDF.
                # OpenBabel -p adjusts ionisation states (COO-, NH3+) without regenerating 3D.
                three_d_path = work_dir / "ligands_3d.sdf"
                three_d_raw_path = work_dir / "ligands_3d_raw.sdf"
                if mols_3d:
                    with Chem.SDWriter(str(three_d_raw_path)) as writer_3d:
                        for mol, name in mols_3d:
                            for prop in list(mol.GetPropsAsDict().keys()):
                                mol.ClearProp(prop)
                            # Strip explicit Hs so obabel -p can add the correct count at
                            # the target pH — keeps heavy-atom 3D coords intact.
                            mol = Chem.RemoveHs(mol, sanitize=False)
                            # Force the title to the resolved name (mol_<i> fallback for
                            # nameless input molecules). Without this an untitled SDF
                            # molecule keeps an empty title through docking, and the PyMOL
                            # session can't build a separate object for it.
                            mol.SetProp('_Name', name)
                            writer_3d.write(mol)
                    # Adjust protonation at target pH (preserves heavy-atom 3D coordinates)
                    ph_proc = await asyncio.create_subprocess_exec(
                        'obabel', str(three_d_raw_path), '-O', str(three_d_path), '-p', str(ph),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        _, ph_stderr = await asyncio.wait_for(ph_proc.communicate(), timeout=120)
                        if ph_proc.returncode != 0 or not three_d_path.exists():
                            logger.warning("OpenBabel pH adjustment for 3D ligands failed: %s — using raw input",
                                           ph_stderr.decode(errors='replace'))
                            three_d_path = three_d_raw_path
                        else:
                            logger.info("OpenBabel pH %.1f adjustment applied to %d 3D molecules", ph, len(mols_3d))
                    except asyncio.TimeoutError:
                        logger.warning("obabel pH adjustment for 3D ligands timed out — using raw input")
                        try:
                            ph_proc.kill()
                            await ph_proc.communicate()
                        except Exception:
                            pass
                        three_d_path = three_d_raw_path

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

                # Combine 3D + prepared-2D into final ligand file
                with open(str(ligand_path), 'w') as out_f:
                    if mols_3d and three_d_path.exists():
                        out_f.write(three_d_path.read_text())
                    if smiles_2d and two_d_prepared_path.exists():
                        out_f.write(two_d_prepared_path.read_text())

            active_jobs[job_id].timings['preparation'] = (datetime.now() - prep_start).total_seconds()

            # Validate final ligand file
            verify_suppl = Chem.SDMolSupplier(str(ligand_path))
            mol_count = sum(1 for mol in verify_suppl if mol is not None)

            if mol_count == 0:
                raise HTTPException(400, "No valid 3D molecules could be prepared from SDF file")

            active_jobs[job_id].total_ligands = mol_count
            logger.info(f"SDF preparation complete: {mol_count} molecules ready for docking")
        
        # Run docking. Flexible docking reuses the reference ligand to choose
        # which side chains flex (flexdist_ligand == autobox_ligand).
        dock_start = datetime.now()
        docked_path, flex_map_path = await job_processor.run_docking_job(
            job_id=job_id,
            receptor_path=str(receptor_path),
            reference_path=str(reference_path) if reference_path else None,
            center=site_center,
            size=site_size,
            ligand_path=str(ligand_path),
            output_dir=str(work_dir),
            num_poses=num_poses,
            exhaustiveness=exhaustiveness,
            cnn_scoring=cnn_scoring,
            seed=seed,
            flexdist_ligand=str(reference_path) if (flexible and reference_path) else None,
            flexdist=flexdist if flexible else None,
            covalent_rec_atom=covalent_rec_atom,
            covalent_lig_atom_pattern=covalent_smarts.strip() if covalent else None,
            covalent_optimize_lig=covalent_optimize,
        )
        active_jobs[job_id].timings['docking'] = (datetime.now() - dock_start).total_seconds()

        if active_jobs[job_id].cancelled:
            raise HTTPException(499, "Job was cancelled")

        # Merge GNINA's split structure/property blocks before any further processing.
        # obabel discards zero-atom property blocks, so fix them first or scores are lost.
        docked_fixed_path = work_dir / "docked_fixed.sdf"
        _raw_gnina = Path(docked_path).read_text(errors='replace')
        docked_fixed_path.write_text(_fix_split_sdf_blocks(_raw_gnina))
        docked_path = str(docked_fixed_path)

        # Restore formal charges stripped by GNINA (e.g. COO⁻, NH₃⁺) via obabel -p.
        # GNINA writes geometrically correct poses but omits M CHG records; without them
        # DataWarrior adds implicit H to undervalenced atoms (COO → COOH).
        docked_ph_path = work_dir / "docked_ph.sdf"
        ph_proc = await asyncio.create_subprocess_exec(
            'obabel', str(docked_path), '-O', str(docked_ph_path), '-p', str(ph),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, ph_stderr = await asyncio.wait_for(ph_proc.communicate(), timeout=120)
            if ph_proc.returncode == 0 and docked_ph_path.exists():
                logger.info("Applied pH %.1f charge correction to GNINA output", ph)
                docked_path = str(docked_ph_path)
            else:
                logger.warning("obabel pH correction on GNINA output failed: %s",
                               ph_stderr.decode(errors='replace'))
        except asyncio.TimeoutError:
            logger.warning("obabel pH correction timed out — using unfixed charges")
            try:
                ph_proc.kill()
                await ph_proc.communicate()
            except Exception:
                pass

        # Sort results
        await job_processor.update_progress(
            job_id,
            status=JobStatus.FINALIZING,
            progress=90,
            message="Sorting and formatting results...",
            current_stage="Finalization"
        )

        final_path = work_dir / "docking_results.sdf"
        loop = asyncio.get_running_loop()
        num_poses_out = await loop.run_in_executor(
            None, job_processor.sort_and_filter_results,
            docked_path, str(final_path), sort_by,
        )

        if num_poses_out == 0:
            raise HTTPException(500, "Docking produced no poses. Check server logs for GPU errors.")

        # Flexible docking: build a pose-ordered PDB of the moved side chains
        # (MODEL N ↔ pose N in docking_results.sdf) for download.
        flex_blocks: Dict[str, str] = {}
        flex_pdb_path: Optional[Path] = None
        if flexible and flex_map_path and os.path.exists(flex_map_path):
            with open(flex_map_path) as _f:
                flex_blocks = json.load(_f)
            flex_pdb_path = work_dir / "flex_residues.pdb"
            n_flex = job_processor.build_flex_pdb(
                final_sdf=str(final_path),
                flex_blocks=flex_blocks,
                output_path=str(flex_pdb_path),
            )
            logger.info(f"Job {job_id}: wrote {n_flex} flex residue models → {flex_pdb_path}")
            if n_flex == 0:
                flex_pdb_path = None

        # Collect postprocessing stats for the run log
        run_log: List[str] = []
        run_log.append(f"GNINA Docking Run Log")
        run_log.append(f"{'=' * 40}")
        run_log.append(f"Session:       {session_name or job_id}")
        run_log.append(f"Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        run_log.append("")
        run_log.append("Parameters")
        run_log.append("-" * 40)
        run_log.append(f"Receptor:      {receptor.filename}")
        if reference_path:
            run_log.append(f"Reference:     {reference.filename if reference else 'n/a'}")
        run_log.append(f"pH:            {ph}{' (skipped — SDF used as-is)' if skip_ligprep else ''}")
        run_log.append(f"Poses:         {num_poses}")
        run_log.append(f"Exhaustiveness:{exhaustiveness}")
        run_log.append(f"CNN scoring:   {cnn_scoring}")
        run_log.append(f"Sort by:       {sort_by}")
        pp_flags = [f for f, en in [
            ("MCS_RMSD", mcs_rmsd), ("Shape_Sim", shape_sim), ("Ref_Sim", ref_sim),
            ("PLIF_Sim", plif_sim), ("PoseBusters", posebusters),
        ] if en]
        run_log.append(f"Postprocessing:{', '.join(pp_flags) if pp_flags else 'none'}")
        run_log.append("")
        run_log.append("Results")
        run_log.append("-" * 40)
        run_log.append(f"Output poses:  {num_poses_out}")

        _executor_loop = asyncio.get_running_loop()

        async def _run_pp(fn, *args, timeout_s=120, label="post-processing"):
            """Run a synchronous post-processing function in the thread pool with a timeout."""
            try:
                return await asyncio.wait_for(
                    _executor_loop.run_in_executor(None, fn, *args),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error("%s timed out after %ds — skipping", label, timeout_s)
                return None

        # Optional MCS RMSD annotation (requires reference ligand)
        if mcs_rmsd and reference_path:
            await job_processor.update_progress(
                job_id, progress=92, message="Calculating MCS RMSD...",
                current_stage="MCS RMSD"
            )
            result = await _run_pp(
                job_processor.add_mcs_rmsd, str(final_path), str(reference_path),
                timeout_s=60, label="MCS RMSD",
            )
            n = result if result is not None else 0
            run_log.append(f"MCS_RMSD:      {n}/{num_poses_out} annotated")

        # Optional shape similarity annotation (requires reference ligand)
        if shape_sim and reference_path:
            await job_processor.update_progress(
                job_id, progress=93, message="Calculating shape similarity...",
                current_stage="Shape Sim"
            )
            result = await _run_pp(
                job_processor.add_shape_sim, str(final_path), str(reference_path),
                timeout_s=60, label="Shape Sim",
            )
            n = result if result is not None else 0
            run_log.append(f"Shape_Sim:     {n}/{num_poses_out} annotated")

        # Optional 2D Morgan ECFP4 similarity to reference
        if ref_sim and reference_path:
            await job_processor.update_progress(
                job_id, progress=93, message="Calculating 2D similarity to reference...",
                current_stage="Ref Sim"
            )
            result = await _run_pp(
                job_processor.add_ref_sim, str(final_path), str(reference_path),
                timeout_s=60, label="Ref Sim",
            )
            n = result if result is not None else 0
            run_log.append(f"Ref_Sim:       {n}/{num_poses_out} annotated")

        if posebusters:
            await job_processor.update_progress(
                job_id, progress=94, message="Running PoseBusters validation...",
                current_stage="PoseBusters"
            )
            result = await _run_pp(
                job_processor.add_posebusters_flags, str(final_path), str(receptor_path),
                timeout_s=300, label="PoseBusters",
            )
            if result is not None:
                n, pb_summary = result
                run_log.append(f"PoseBusters:   {n}/{num_poses_out} evaluated (dock mode)")
                if pb_summary:
                    run_log.append("")
                    run_log.append("PoseBusters failure counts across all poses:")
                    for line in pb_summary.splitlines():
                        run_log.append(f"  {line}")
            else:
                run_log.append(f"PoseBusters:   timed out — skipped")

        if plif_sim and reference_path:
            await job_processor.update_progress(
                job_id, progress=95, message="Calculating PLIF similarity...",
                current_stage="PLIF Sim"
            )
            result = await _run_pp(
                job_processor._add_plif_sim_sync,
                str(final_path), str(receptor_path), str(reference_path),
                timeout_s=300, label="PLIF Sim",
            )
            n = result if result is not None else 0
            run_log.append(f"PLIF_Sim:      {n}/{num_poses_out} annotated")

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
                reference_path=str(reference_path) if reference_path else None,
                center=site_center,
                size=site_size,
                docking_results_sdf=str(final_path),
                session_name=session_name,
                sort_by=sort_by,
                flex_blocks=flex_blocks if flexible else None,
            )

        total_time = (datetime.now() - start_time).total_seconds()
        active_jobs[job_id].timings['total'] = total_time

        run_log.append("")
        run_log.append("Timings")
        run_log.append("-" * 40)
        timings = active_jobs[job_id].timings
        if 'preparation' in timings:
            run_log.append(f"Preparation:   {timings['preparation']:.1f}s")
        if 'docking' in timings:
            run_log.append(f"Docking:       {timings['docking']:.1f}s")
        run_log.append(f"Total:         {total_time:.1f}s")

        await job_processor.update_progress(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            message=f"Completed! {num_poses_out} poses in {total_time:.1f}s"
                    + (" (+ PyMOL session)" if pse_path else ""),
            result_file=str(final_path)
        )
        logger.warning(f"Job {job_id} completed: {num_poses_out} poses in {total_time:.1f}s")
        _client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
        _n_ligands = active_jobs[job_id].total_ligands or 0
        access_logger.info(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ip={_client_ip} | "
            f"job={job_id} | ligands={_n_ligands} | poses={num_poses_out} | time={total_time:.1f}s"
        )

        stem = session_name if session_name else f"docking_{job_id}"

        # Write per-job run log and always bundle everything into a ZIP.
        log_path = work_dir / f"{stem}_run.log"
        log_path.write_text('\n'.join(run_log) + '\n')

        zip_path = work_dir / f"{stem}.zip"
        with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(final_path), f"{stem}.sdf")
            zf.write(str(log_path), f"{stem}_run.log")
            if pse_path:
                zf.write(pse_path, f"{stem}.pse")
            if flex_pdb_path:
                zf.write(str(flex_pdb_path), f"{stem}_flex.pdb")
        active_jobs[job_id].result_zip = str(zip_path)

        resp = FileResponse(
            path=str(zip_path),
            filename=f"{stem}.zip",
            media_type="application/zip",
        )
        resp.headers["X-Job-Id"] = job_id
        return resp
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        active_jobs[job_id].status = JobStatus.FAILED
        active_jobs[job_id].error = str(e)
        raise HTTPException(500, f"Docking failed: {str(e)}")
    
    finally:
        async def cleanup():
            failed = active_jobs.get(job_id, {}) and active_jobs[job_id].status == JobStatus.FAILED
            await asyncio.sleep(300 if failed else 86400)  # 5 min for failures, 24 h for success
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            active_jobs.pop(job_id, None)
            job_websockets.pop(job_id, None)

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


@app.get("/jobs/{job_id}/download")
async def download_job_results(job_id: str, session_name: str = ""):
    """Re-download results for a completed job (available for up to 24 hours)."""
    if job_id not in active_jobs:
        raise HTTPException(404, "Job not found or already expired")
    job = active_jobs[job_id]
    if job.status != JobStatus.COMPLETED or not job.result_file:
        raise HTTPException(400, "Job not completed or no results available")
    result_path = Path(job.result_file)
    if not result_path.exists():
        raise HTTPException(404, "Result file no longer available")
    # Prefer the bundled zip (SDF + PyMOL session) recorded on the job.
    if job.result_zip:
        zip_path = Path(job.result_zip)
        if zip_path.exists():
            return FileResponse(path=str(zip_path), filename=zip_path.name, media_type="application/zip")
    stem = session_name if session_name else f"docking_{job_id}"
    return FileResponse(path=str(result_path), filename=f"{stem}.sdf", media_type="application/octet-stream")


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


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running docking job by killing its GNINA subprocesses."""
    job = active_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(400, "Job already finished")
    job.cancelled = True
    for proc in job.gnina_procs:
        try:
            proc.terminate()
        except Exception:
            pass
    await job_processor.update_progress(
        job_id,
        status=JobStatus.FAILED,
        message="Cancelled by user",
    )
    logger.warning(f"Job {job_id} cancelled by user")
    return {"status": "cancelled"}


# ============================================================================
# PROTEIN PREPARATION ENDPOINTS
# ============================================================================

@app.post("/protprep/inspect")
async def protprep_inspect(
    pdb_file: Optional[UploadFile] = File(None),
    pdb_id: Optional[str] = Form(None),
):
    """
    Step 1: Load a PDB (upload or fetch by ID) and return structural info
    including all HETATM groups so the user can pick a reference ligand.
    """
    if not pdb_file and not (pdb_id and pdb_id.strip()):
        raise HTTPException(400, "Provide pdb_file or pdb_id")

    token = str(uuid.uuid4())[:8]
    prep_dir = WORK_DIR / f"protprep_{token}"
    prep_dir.mkdir(parents=True, exist_ok=True)

    try:
        if pdb_id and pdb_id.strip():
            pdb_id = pdb_id.strip().upper()
            if not re.match(r'^[A-Z0-9]{4}$', pdb_id):
                raise HTTPException(400, f"Invalid PDB ID: {pdb_id!r}")
            pdb_path = prep_dir / f"{pdb_id}.pdb"
            # Fetch PDB; fall back to mmCIF → PDB via PDBFixer for structures
            # not available in legacy PDB format (e.g. 6RLW, large assemblies)
            fetch_script = prep_dir / "_fetch.py"
            fetch_script.write_text(
                f"import urllib.request, urllib.error\n"
                f"from pathlib import Path\n"
                f"out = Path({repr(str(pdb_path))})\n"
                f"pdb_id = {repr(pdb_id)}\n"
                f"try:\n"
                f"    urllib.request.urlretrieve(\n"
                f"        f'https://files.rcsb.org/download/{{pdb_id}}.pdb', str(out))\n"
                f"except urllib.error.HTTPError as e:\n"
                f"    if e.code != 404:\n"
                f"        raise\n"
                f"    cif_tmp = out.with_suffix('.cif')\n"
                f"    urllib.request.urlretrieve(\n"
                f"        f'https://files.rcsb.org/download/{{pdb_id}}.cif', str(cif_tmp))\n"
                f"    from pdbfixer import PDBFixer\n"
                f"    from openmm.app import PDBFile\n"
                f"    fixer = PDBFixer(filename=str(cif_tmp))\n"
                f"    with open(str(out), 'w') as f:\n"
                f"        PDBFile.writeFile(fixer.topology, fixer.positions, f)\n"
                f"    cif_tmp.unlink()\n"
            )
            proc = await asyncio.create_subprocess_exec(
                OPENMMDL_PYTHON, str(fetch_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.communicate()
                except Exception:
                    pass
                raise HTTPException(504, f"Timed out fetching {pdb_id} from RCSB")
            if proc.returncode != 0 or not pdb_path.exists():
                raise HTTPException(400, f"Could not fetch {pdb_id} from RCSB: {stderr.decode()[:300]}")
        else:
            content = await pdb_file.read()
            pdb_path = prep_dir / secure_filename(pdb_file.filename)
            pdb_path.write_bytes(content)

        # Run _inspect via openmmdl python — silences all rich output
        inspect_script = prep_dir / "_inspect.py"
        inspect_script.write_text(
            f"import sys, json\n"
            f"sys.path.insert(0, {repr(str(Path(PROTPREP_SCRIPT).parent))})\n"
            f"import protprep\n"
            f"_noop = lambda *a, **k: None\n"
            f"for _fn in ['_print','_ok','_warn','_info','_err','_step','_header','_rule','_rule']:\n"
            f"    setattr(protprep, _fn, _noop)\n"
            f"protprep._fatal = lambda msg: (_ for _ in ()).throw(SystemExit(msg))\n"
            f"from pathlib import Path\n"
            f"info = protprep._inspect(Path({repr(str(pdb_path))}))\n"
            f"print(json.dumps(info))\n"
        )
        proc = await asyncio.create_subprocess_exec(
            OPENMMDL_PYTHON, str(inspect_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            raise HTTPException(504, "PDB inspection timed out")
        if proc.returncode != 0:
            raise HTTPException(500, f"Inspect failed: {stderr.decode()[:500]}")

        info = json.loads(stdout.decode())
        return {
            "token": token,
            "pdb_filename": pdb_path.name,
            "chains": info["chains"],
            "chain_info": info["chain_info"],
            "n_std": info["n_std"],
            "n_water": info["n_water"],
            "het_groups": info["het_groups"],
            "n_altloc": info["n_altloc"],
            "ssbonds": len(info["ssbonds"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(prep_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


@app.post("/protprep/run")
async def protprep_run(
    token: str = Form(...),
    keep_het: Optional[str] = Form(None),
    ph: float = Form(7.4),
    chains: Optional[str] = Form(None),    # space-separated, blank = all
    cofactors: Optional[str] = Form(None), # space-separated resnames to keep in receptor
):
    """
    Step 2: Run the full protein preparation pipeline and return
    the prepared receptor PDB and extracted reference ligand SDF as base64.
    """
    if not re.match(r'^[a-f0-9]{8}$', token):
        raise HTTPException(400, "Invalid session token")
    prep_dir = WORK_DIR / f"protprep_{token}"
    if not prep_dir.exists():
        raise HTTPException(404, "Session not found — please inspect the protein again")

    pdb_files = [f for f in prep_dir.glob("*.pdb") if not f.stem.startswith("_")]
    if not pdb_files:
        raise HTTPException(404, "PDB file not found in session")
    input_pdb = pdb_files[0]
    stem = input_pdb.stem
    output_pdb = prep_dir / f"{stem}_prepared.pdb"

    cmd = [
        OPENMMDL_PYTHON, PROTPREP_SCRIPT,
        "--input",    str(input_pdb),
        "--output",   str(output_pdb),
        "--ph",       str(ph),
        "--no-pdb2pqr",  # pdb2pqr not installed; PDBFixer used instead
        "--minimize",    # OpenMM restrained vacuum minimization to optimize H positions
    ]
    if keep_het and keep_het.strip():
        cmd += ["--keep-het", keep_het]
    if chains and chains.strip():
        cmd += ["--chain"] + chains.split()
    if cofactors and cofactors.strip():
        cmd += ["--cofactor"] + cofactors.split()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(prep_dir),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.communicate()
        except Exception:
            pass
        raise HTTPException(504, "Protein preparation timed out (>10 min)")
    log_output = stdout.decode() + "\n" + stderr.decode()

    if proc.returncode != 0 or not output_pdb.exists():
        raise HTTPException(500, f"Protein preparation failed:\n{log_output[-2000:]}")

    # --minimize writes a separate _minimized.pdb; prefer it over _prepared.pdb
    # since the prepared file is written before minimization runs.
    minimized_pdb = prep_dir / f"{stem}_minimized.pdb"
    final_pdb = minimized_pdb if minimized_pdb.exists() else output_pdb

    prepared_pdb_b64 = base64.b64encode(final_pdb.read_bytes()).decode()

    lig_sdf = prep_dir / f"{stem}_prepared_ligand.sdf"
    ligand_sdf_b64 = base64.b64encode(lig_sdf.read_bytes()).decode() if lig_sdf.exists() else None

    return {
        "prepared_pdb_name": final_pdb.name,
        "prepared_pdb_b64": prepared_pdb_b64,
        "ligand_sdf_name": lig_sdf.name if lig_sdf.exists() else None,
        "ligand_sdf_b64": ligand_sdf_b64,
        "log": log_output[-3000:],
    }


if __name__ == "__main__":
    uvicorn.run(
        "gnina_webapp:app",
        host=os.environ.get("BIND_HOST", "0.0.0.0"),
        port=int(os.environ.get("BIND_PORT", "5004")),
        workers=1,  # Single worker for shared state
        reload=False,
        log_level="info"
    )
