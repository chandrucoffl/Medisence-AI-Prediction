// MediSense AI — Aurora Bento ambient effects.
// Floating particles + mouse-tracking card tilt + animated count-up
// stats + a confetti burst on good news. Purely decorative, and every
// effect respects prefers-reduced-motion.
(function () {
  const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ── 1. Floating ambient particles ──
  if (!REDUCED) {
    const COLORS = ['#2EEAC0', '#22D3EE', '#A78BFA', '#FB7BC4', '#FFB020'];
    const COUNT = 16;
    for (let i = 0; i < COUNT; i++) {
      const p = document.createElement('div');
      p.className = 'particle';
      const size = 2 + Math.random() * 3;
      const left = Math.random() * 100;
      const duration = 10 + Math.random() * 10;
      const delay = Math.random() * 12;
      const color = COLORS[Math.floor(Math.random() * COLORS.length)];
      p.style.width = size + 'px';
      p.style.height = size + 'px';
      p.style.left = left + 'vw';
      p.style.background = color;
      p.style.boxShadow = `0 0 ${size * 3}px ${size}px ${color}`;
      p.style.animationDuration = duration + 's';
      p.style.animationDelay = delay + 's';
      document.body.appendChild(p);
    }
  }

  // ── 2. Mouse-tracking 3D tilt on cards ──
  if (!REDUCED && window.matchMedia('(hover: hover)').matches) {
    const tiltables = document.querySelectorAll('.module-card, .card, .verdict-panel');
    tiltables.forEach(el => {
      el.addEventListener('mousemove', e => {
        const r = el.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width;
        const y = (e.clientY - r.top) / r.height;
        const rotY = (x - 0.5) * 8;
        const rotX = (0.5 - y) * 8;
        el.style.transform = `perspective(900px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(-4px)`;
        el.style.setProperty('--mx', (x * 100) + '%');
        el.style.setProperty('--my', (y * 100) + '%');
      });
      el.addEventListener('mouseleave', () => { el.style.transform = ''; });
    });
  }

  // ── 3. Animated count-up for numeric readouts (e.g. "97.4%", "06") ──
  function animateCount(el) {
    const raw = el.textContent.trim();
    const match = raw.match(/^(\d+(?:\.\d+)?)(.*)$/);
    if (!match) return;
    const target = parseFloat(match[1]);
    const suffix = match[2] || '';
    const decimals = (match[1].split('.')[1] || '').length;
    const duration = 900;
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = raw;
    }
    if (REDUCED) return;
    requestAnimationFrame(tick);
  }
  document.querySelectorAll('.readout .val').forEach(animateCount);

  // ── 4. Confetti burst on a "Low Risk" / good-news verdict ──
  function confettiBurst() {
    if (REDUCED) return;
    const COLORS = ['#2EEAC0', '#22D3EE', '#A78BFA', '#FB7BC4', '#FFB020'];
    for (let i = 0; i < 40; i++) {
      const c = document.createElement('div');
      const size = 5 + Math.random() * 6;
      const color = COLORS[Math.floor(Math.random() * COLORS.length)];
      c.style.position = 'fixed';
      c.style.left = '50%';
      c.style.top = '18%';
      c.style.width = size + 'px';
      c.style.height = size + 'px';
      c.style.background = color;
      c.style.borderRadius = Math.random() > 0.5 ? '50%' : '3px';
      c.style.zIndex = 999;
      c.style.pointerEvents = 'none';
      const angle = Math.random() * Math.PI * 2;
      const dist = 120 + Math.random() * 260;
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist + 80;
      c.animate([
        { transform: 'translate(0,0) rotate(0deg)', opacity: 1 },
        { transform: `translate(${dx}px, ${dy}px) rotate(${360 + Math.random()*360}deg)`, opacity: 0 }
      ], { duration: 1100 + Math.random() * 500, easing: 'cubic-bezier(.2,.8,.3,1)' });
      document.body.appendChild(c);
      setTimeout(() => c.remove(), 1700);
    }
  }
  if (document.querySelector('.verdict-panel.low')) {
    setTimeout(confettiBurst, 250);
  }
})();
