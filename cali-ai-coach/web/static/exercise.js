// Exercise page: HUD poll + phone camera (getUserMedia) + TTS + reference switching
(function() {
  var scoreEl = document.getElementById('hud-score');
  var repsEl = document.getElementById('hud-reps');
  var statScore = document.getElementById('stat-score');
  var statReps = document.getElementById('stat-reps');
  var statCp = document.getElementById('stat-cp');
  var errorList = document.getElementById('error-list');
  var feedbackList = document.getElementById('feedback-list');
  var statusText = document.getElementById('status-text');
  var ttsBar = document.getElementById('tts-bar');
  var camFeed = document.getElementById('cam-feed');
  var refSelect = document.getElementById('ref-source');

  var lastTts = '';
  var ttsCooldown = 0;

  function speak(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.lang = 'vi-VN';
    u.rate = 0.9;
    speechSynthesis.speak(u);
  }

  function updateUI(data) {
    var score = data.score || 0;
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
      var latest = data.feedbacks[0];
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

  // ---- Debug log helper ----
  var debugEl = document.getElementById('camera-debug');
  function log(msg) { console.log('[cam]', msg); if (debugEl) debugEl.textContent = new Date().toLocaleTimeString() + ' ' + msg; }

  // ---- Camera: each device uses its own local camera via getUserMedia ----
  log('init: starting...');
  (function() {
    var video = document.getElementById('phone-cam');
    var canvas = document.getElementById('phone-canvas');
    var ctx = canvas.getContext('2d');
    var sending = false;
    var lastBlobUrl = null;

    // Check MediaDevices API availability
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      log('FAIL: navigator.mediaDevices.getUserMedia not available (insecure origin?)');
      camFeed.src = '/video_feed?exercise=' + encodeURIComponent(EXERCISE);
      log('fallback: MJPEG stream');
      return;
    }

    function sendFrame() {
      if (!video.videoWidth) { requestAnimationFrame(sendFrame); return; }
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      canvas.toBlob(function(blob) {
        if (!blob) { log('sendFrame: blob null, retry'); requestAnimationFrame(sendFrame); return; }
        var fd = new FormData();
        fd.append('frame', blob, 'frame.jpg');
        fd.append('exercise', EXERCISE);
        sending = true;
        fetch('/process_frame', { method: 'POST', body: fd })
          .then(function(r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.blob();
          })
          .then(function(imgBlob) {
            if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
            lastBlobUrl = URL.createObjectURL(imgBlob);
            camFeed.src = lastBlobUrl;
            sending = false;
            requestAnimationFrame(sendFrame);
          })
          .catch(function(e) {
            log('sendFrame error: ' + e.message);
            sending = false;
            setTimeout(sendFrame, 2000);
          });
      }, 'image/jpeg', 0.5);
    }

    log('requesting getUserMedia...');
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } })
      .then(function(stream) {
        log('getUserMedia OK');
        video.srcObject = stream;
        video.onloadedmetadata = function() {
          log('video metadata loaded: ' + video.videoWidth + 'x' + video.videoHeight);
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          video.style.display = 'none';
          camFeed.style.display = 'block';
          camFeed.style.width = '100%';
          camFeed.style.height = '100%';
          camFeed.style.objectFit = 'cover';
          sendFrame();
        };
        video.onerror = function(e) { log('video element error: ' + (e.message || 'unknown')); };
      })
      .catch(function(err) {
        log('getUserMedia FAILED: ' + err.message + ' (' + err.name + ')');
        log('fallback: MJPEG stream');
        camFeed.src = '/video_feed?exercise=' + encodeURIComponent(EXERCISE);
        camFeed.style.display = 'block';
      });
  })();

  // ---- Poll HUD ----
  setInterval(async function() {
    try {
      var r = await fetch('/api/status');
      updateUI(await r.json());
    } catch(e) {}
  }, 400);

  // ---- Reference selector ----
  if (refSelect) {
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
      if (val !== EXERCISE) {
        window.location.href = '/exercise/' + encodeURIComponent(val);
      }
    });
  }
})();
