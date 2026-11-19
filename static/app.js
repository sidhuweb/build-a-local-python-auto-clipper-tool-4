// AutoClipper Frontend JavaScript

let currentJobId = null;
let pollTimer = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  checkDemoAvailability();

  document.getElementById("btn-clip").addEventListener("click", startClipping);
  document.getElementById("btn-demo").addEventListener("click", loadDemoMode);
});

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const card = e.target.closest(".clip-card");
      if (!card) return;
      const targetTab = e.target.getAttribute("data-tab");

      card.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      card.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      e.target.classList.add("active");
      const targetPane = card.querySelector(`.tab-pane[data-pane="${targetTab}"]`);
      if (targetPane) targetPane.classList.add("active");
    });
  });
}

async function checkDemoAvailability() {
  try {
    const res = await fetch("/api/demo-clips");
    if (res.ok) {
      const data = await res.json();
      if (data.clips && data.clips.length > 0) {
        renderResults(data);
        return;
      }
    }
  } catch (err) {
    console.log("API not available, trying static file...");
  }

  // Fallback for static preview mode
  try {
    const resStatic = await fetch("/static/clips/all_clips_summary.json");
    if (resStatic.ok) {
      const data = await resStatic.json();
      (data.clips || []).forEach(c => {
        c.video_url = `/static/clips/${c.relative_video_file}`;
        c.metadata_url = `/static/clips/${c.relative_metadata_file}`;
      });
      renderResults(data);
    }
  } catch (err2) {
    console.log("Static clips summary not loaded:", err2);
  }
}

async function startClipping() {
  const urlInput = document.getElementById("input-url");
  const url = urlInput.value.trim();

  if (!url) {
    alert("Please enter a YouTube video/stream URL or local video path!");
    urlInput.focus();
    return;
  }

  const duration = parseFloat(document.getElementById("select-duration").value) || 45.0;
  const facecam = document.getElementById("select-facecam").value || "top-right";
  const model = document.getElementById("select-model").value || "tiny";
  const color = document.getElementById("select-color").value || "&H0000FFFF";

  showProgress(true);
  updateProgress(5, "Connecting to AutoClipper engine...");

  try {
    const res = await fetch("/api/clip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: url,
        clip_duration: duration,
        facecam_preset: facecam,
        whisper_model: model,
        highlight_color: color,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to start clipping");
    }

    const data = await res.json();
    currentJobId = data.job_id;
    startPolling(currentJobId);

  } catch (err) {
    console.error("Error starting clip:", err);
    updateProgress(0, "Error: " + err.message);
    setTimeout(() => {
      alert("AutoClipper Error: " + err.message);
      showProgress(false);
    }, 1500);
  }
}

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${jobId}`);
      if (!res.ok) return;

      const job = await res.json();
      updateProgress(job.progress, job.stage);

      if (job.status === "completed") {
        clearInterval(pollTimer);
        setTimeout(() => {
          showProgress(false);
          renderResults(job);
        }, 800);
      } else if (job.status === "failed") {
        clearInterval(pollTimer);
        alert("Clipping failed: " + job.error);
        showProgress(false);
      }
    } catch (e) {
      console.log("Polling error:", e);
    }
  }, 1000);
}

function showProgress(visible) {
  const panel = document.getElementById("progress-panel");
  panel.style.display = visible ? "block" : "none";
  if (visible) {
    panel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function updateProgress(pct, stage) {
  document.getElementById("progress-fill").style.width = pct + "%";
  document.getElementById("progress-percent").textContent = Math.round(pct) + "%";
  document.getElementById("progress-stage").textContent = stage;
}

async function loadDemoMode() {
  document.getElementById("input-url").value = "samples/sample_stream.mp4";
  showProgress(true);
  updateProgress(20, "Loading realistic streamer sample demo...");

  try {
    const res = await fetch("/api/demo-clips");
    const data = await res.json();

    if (data.clips && data.clips.length > 0) {
      updateProgress(100, "Demo loaded successfully!");
      setTimeout(() => {
        showProgress(false);
        renderResults(data);
      }, 500);
    } else {
      // Trigger via API
      startClipping();
    }
  } catch (e) {
    startClipping();
  }
}

function renderResults(data) {
  const clips = data.clips || [];
  const timeline = data.timeline || [];

  if (timeline.length > 0) {
    renderTimelineChart(timeline, clips);
    document.getElementById("timeline-section").style.display = "block";
  }

  const grid = document.getElementById("clips-grid");
  grid.innerHTML = "";

  if (clips.length === 0) {
    grid.innerHTML = "<p style='color: var(--text-muted); text-align: center; width: 100%;'>No clips found. Paste a video URL and click Auto-Clip!</p>";
    return;
  }

  clips.forEach((clip, idx) => {
    const card = createClipCard(clip, idx + 1);
    grid.appendChild(card);
  });

  initTabs();
  document.getElementById("clips-section").style.display = "block";
  grid.scrollIntoView({ behavior: "smooth", block: "start" });
}

function createClipCard(clip, num) {
  const card = document.createElement("div");
  card.className = "clip-card";

  const videoUrl = clip.video_url || `/static/clips/${clip.relative_video_file}` || `output/test_run/${clip.relative_video_file}`;
  const metadataUrl = clip.metadata_url || `/static/clips/${clip.relative_metadata_file}` || `output/test_run/${clip.relative_metadata_file}`;

  const altTitles = (clip.alternative_titles || []).map((t, i) => `<li>${t}</li>`).join("");

  card.innerHTML = `
    <div class="clip-header-badge">VIRAL MOMENT #${clip.rank || num}</div>
    <div class="clip-viral-score">⚡ SCORE ${clip.viral_score || 95}/100</div>

    <div class="video-wrapper-916">
      <video controls playsinline preload="metadata">
        <source src="${videoUrl}" type="video/mp4">
        Your browser does not support HTML5 video.
      </video>
    </div>

    <div class="clip-content">
      <h3 class="clip-title">${clip.title}</h3>
      <div class="clip-meta-row">
        <span>⏱️ ${clip.duration}s (${clip.start_time}s - ${clip.end_time}s)</span>
      </div>
      <div class="clip-reason-tag">🔥 ${clip.primary_reason || "Audio Spike + Replay Peak"}</div>

      <div class="tab-row">
        <button class="tab-btn active" data-tab="desc">Description</button>
        <button class="tab-btn" data-tab="tags">Hashtags</button>
        <button class="tab-btn" data-tab="titles">A/B Titles</button>
        <button class="tab-btn" data-tab="trans">Transcript</button>
      </div>

      <div class="tab-pane active" data-pane="desc">
        <div class="copy-box">
          <button class="copy-btn-floating" onclick="copyText(this)">Copy</button>
          <span>${escapeHtml(clip.description || "")}</span>
        </div>
      </div>

      <div class="tab-pane" data-pane="tags">
        <div class="copy-box">
          <button class="copy-btn-floating" onclick="copyText(this)">Copy</button>
          <span>${clip.hashtags_string || (clip.hashtags || []).join(" ")}</span>
        </div>
      </div>

      <div class="tab-pane" data-pane="titles">
        <div class="copy-box">
          <button class="copy-btn-floating" onclick="copyText(this)">Copy</button>
          <ul style="padding-left: 18px; margin: 0;">${altTitles || "<li>" + clip.title + "</li>"}</ul>
        </div>
      </div>

      <div class="tab-pane" data-pane="trans">
        <div class="copy-box">
          <button class="copy-btn-floating" onclick="copyText(this)">Copy</button>
          <span>"${escapeHtml(clip.transcript || "")}"</span>
        </div>
      </div>

      <div class="card-actions">
        <a href="${videoUrl}" download="${clip.relative_video_file || 'clip_9x16.mp4'}" class="btn-download-mp4">
          📥 Download 9:16 MP4
        </a>
        <a href="${metadataUrl}" download="${clip.relative_metadata_file || 'metadata.txt'}" class="btn-download-txt">
          📄 Text Kit
        </a>
      </div>
    </div>
  `;
  return card;
}

function renderTimelineChart(timeline, clips) {
  const container = document.getElementById("chart-container");
  if (!container || timeline.length === 0) return;

  const width = container.clientWidth || 900;
  const height = 180;
  const padBottom = 24;
  const padTop = 14;

  const maxTime = timeline[timeline.length - 1].time;
  const xScale = (t) => (t / maxTime) * width;
  const yScale = (v) => height - padBottom - (v / 100) * (height - padTop - padBottom);

  // Build SVG Paths
  let audioPath = "";
  let replayPath = "";
  let viralPath = "";

  timeline.forEach((pt, i) => {
    const x = xScale(pt.time);
    const yAudio = yScale(pt.audio);
    const yReplay = yScale(pt.replay);
    const yViral = yScale(pt.viral_score);

    if (i === 0) {
      audioPath += `M ${x} ${yAudio}`;
      replayPath += `M ${x} ${yReplay}`;
      viralPath += `M ${x} ${yViral}`;
    } else {
      audioPath += ` L ${x} ${yAudio}`;
      replayPath += ` L ${x} ${yReplay}`;
      viralPath += ` L ${x} ${yViral}`;
    }
  });

  // Highlight clips boxes on timeline
  let clipBoxes = "";
  (clips || []).forEach((c, idx) => {
    const x1 = xScale(c.start_time);
    const x2 = xScale(c.end_time);
    const w = Math.max(12, x2 - x1);
    clipBoxes += `
      <rect x="${x1}" y="6" width="${w}" height="${height - padBottom - 6}" fill="rgba(0, 240, 255, 0.15)" stroke="#00f0ff" stroke-width="1.5" stroke-dasharray="4,2" rx="4"/>
      <text x="${x1 + 4}" y="20" fill="#00f0ff" font-size="11" font-weight="bold">#${idx + 1}</text>
    `;
  });

  container.innerHTML = `
    <svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="viralGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#ffe600" stop-opacity="0.4"/>
          <stop offset="100%" stop-color="#ffe600" stop-opacity="0.0"/>
        </linearGradient>
      </defs>
      <!-- Grid lines -->
      <line x1="0" y1="${height - padBottom}" x2="${width}" y2="${height - padBottom}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
      <line x1="0" y1="${yScale(50)}" x2="${width}" y2="${yScale(50)}" stroke="rgba(255,255,255,0.05)" stroke-width="1" stroke-dasharray="4"/>
      
      <!-- Curves -->
      <path d="${replayPath}" fill="none" stroke="#9d4edd" stroke-width="1.8" opacity="0.75"/>
      <path d="${audioPath}" fill="none" stroke="#00f0ff" stroke-width="1.8" opacity="0.75"/>
      <path d="${viralPath}" fill="none" stroke="#ffe600" stroke-width="2.5"/>

      <!-- Clip Range Overlays -->
      ${clipBoxes}
    </svg>
  `;
}

function copyText(button) {
  const box = button.nextElementSibling;
  const text = box.innerText || box.textContent;
  navigator.clipboard.writeText(text).then(() => {
    const orig = button.textContent;
    button.textContent = "Copied!";
    button.style.background = "#00ff88";
    button.style.color = "#000";
    setTimeout(() => {
      button.textContent = orig;
      button.style.background = "";
      button.style.color = "";
    }, 1500);
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
