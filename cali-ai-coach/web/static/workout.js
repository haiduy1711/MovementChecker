// Workout page: poll /api/status + update HUD + TTS
(function() {
  const scoreEl = document.getElementById('hud-score');
  const repsEl = document.getElementById('hud-reps');
  const statScore = document.getElementById('stat-score');
  const statReps = document.getElementById('stat-reps');
  const statCp = document.getElementById('stat-cp');
  const errorList = document.getElementById('error-list');
  const feedbackList = document.getElementById('feedback-list');
  const statusText = document.getElementById('status-text');
  const ttsBar = document.getElementById('tts-bar');

  let lastTts = '';
  let ttsCooldown = 0;

  function speak(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'vi-VN';
    u.rate = 0.9;
    speechSynthesis.speak(u);
  }

  function updateUI(data) {
    // Score
    const score = data.score || 0;
    scoreEl.textContent = score;
    statScore.textContent = score;
    scoreEl.className = 'hud-score' + (score >= 80 ? ' good' : score >= 50 ? ' warn' : ' bad');
    statScore.className = 'stat-num' + (score >= 80 ? ' good' : score >= 50 ? ' warn' : ' bad');

    // Reps
    repsEl.textContent = data.reps || 0;
    statReps.textContent = data.reps || 0;

    // Checkpoint
    statCp.textContent = data.cp + '/' + data.total_cp;

    // Status
    statusText.textContent = data.status || '...';

    // Errors
    if (data.errors && data.errors.length) {
      errorList.innerHTML = data.errors.map(e => '<li class="error-item">' + e + '</li>').join('');
    } else {
      errorList.innerHTML = '<li class="no-error">✓ Không có lỗi</li>';
    }

    // Feedback
    if (data.feedbacks && data.feedbacks.length) {
      feedbackList.innerHTML = data.feedbacks.map(f => '<li class="fb-item">' + f + '</li>').join('');
      // TTS
      const latest = data.feedbacks[0];
      if (latest !== lastTts && Date.now() > ttsCooldown) {
        lastTts = latest;
        ttsCooldown = Date.now() + 3000;
        speak(latest);
        ttsBar.textContent = latest;
      }
    } else {
      const msgs = {
        'WAIT': 'Đang giữ checkpoint',
        'PASS': 'Chuẩn!',
        'REP': 'Hoàn thành!',
        'INIT': 'Bắt đầu',
        'CAN CHINH': 'Hãy căn chỉnh sao cho khớp với khung xương ảo',
      };
      const stat = (data.status || '');
      let tip = 'Tốt!';
      for (const [k, v] of Object.entries(msgs)) {
        if (stat.includes(k)) { tip = v; break; }
      }
      feedbackList.innerHTML = '<li class="no-error">' + tip + '</li>';
      if (stat.includes('PASS') || stat.includes('REP')) {
        ttsBar.textContent = '✓ ' + tip;
      } else {
        ttsBar.textContent = tip;
      }
    }
  }

  // Poll every 400ms
  setInterval(async () => {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      updateUI(data);
    } catch (e) {
      // ignore
    }
  }, 400);
})();
