// Exercise page: poll /api/status + update HUD + TTS + reference switching
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
  const camFeed = document.getElementById('cam-feed');
  const refSelect = document.getElementById('ref-source');

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
    const score = data.score || 0;
    scoreEl.textContent = score;
    statScore.textContent = score;
    scoreEl.className = 'hud-score' + (score >= 80 ? ' good' : score >= 50 ? ' warn' : ' bad');
    statScore.className = 'stat-num' + (score >= 80 ? ' good' : score >= 50 ? ' warn' : ' bad');

    repsEl.textContent = data.reps || 0;
    statReps.textContent = data.reps || 0;
    statCp.textContent = data.cp + '/' + data.total_cp;
    statusText.textContent = data.status || '...';

    if (data.errors && data.errors.length) {
      errorList.innerHTML = data.errors.map(function(e) { return '<li class="error-item">' + e + '</li>'; }).join('');
    } else {
      errorList.innerHTML = '<li class="no-error">✓ Không có lỗi</li>';
    }

    if (data.feedbacks && data.feedbacks.length) {
      feedbackList.innerHTML = data.feedbacks.map(function(f) { return '<li class="fb-item">' + f + '</li>'; }).join('');
      const latest = data.feedbacks[0];
      if (latest !== lastTts && Date.now() > ttsCooldown) {
        lastTts = latest;
        ttsCooldown = Date.now() + 3000;
        speak(latest);
        ttsBar.textContent = latest;
      }
    } else {
      var msgs = {
        'WAIT': 'Đang giữ checkpoint',
        'PASS': 'Chuẩn!',
        'REP': 'Hoàn thành!',
        'INIT': 'Bắt đầu',
        'CAN CHINH': 'Hãy căn chỉnh sao cho khớp với khung xương ảo màu vàng',
      };
      var stat = (data.status || '');
      var tip = 'Tốt!';
      for (var k in msgs) {
        if (stat.includes(k)) { tip = msgs[k]; break; }
      }
      feedbackList.innerHTML = '<li class="no-error">' + tip + '</li>';
      ttsBar.textContent = (stat.includes('PASS') || stat.includes('REP')) ? '✓ ' + tip : tip;
    }
  }

  // Poll
  setInterval(async function() {
    try {
      var r = await fetch('/api/status');
      updateUI(await r.json());
    } catch(e) {}
  }, 400);

  // Reference selector: rebuild feed when changed
  if (refSelect) {
    // Load available exercises
    fetch('/api/exercises').then(function(r) { return r.json(); }).then(function(list) {
      list.forEach(function(ex) {
        if (ex.checkpoints || ex.custom) {
          var opt = document.createElement('option');
          opt.value = ex.id;
          opt.textContent = ex.name;
          if (ex.id === EXERCISE) opt.selected = true;
          refSelect.appendChild(opt);
        }
      });
    });

    refSelect.addEventListener('change', function() {
      var val = refSelect.value;
      camFeed.src = '/video_feed?exercise=' + encodeURIComponent(val);
      ttsBar.textContent = 'Đã chuyển nguồn: ' + val;
      document.getElementById('exercise-title').textContent = val;
    });
  }
})();
