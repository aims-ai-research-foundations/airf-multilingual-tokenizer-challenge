const navLinks = [...document.querySelectorAll('.nav-tab')];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);

function setActiveSection(id) {
  navLinks.forEach((link) => {
    const active = link.getAttribute('href') === `#${id}`;
    link.classList.toggle('active', active);
    if (active) {
      link.setAttribute('aria-current', 'true');
      link.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
    } else {
      link.removeAttribute('aria-current');
    }
  });
}

let scrollTicking = false;
function syncNavigation() {
  const marker = window.scrollY + Math.min(240, window.innerHeight * 0.36);
  let current = sections[0];
  sections.forEach((section) => {
    if (section.offsetTop <= marker) current = section;
  });
  if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
    current = sections[sections.length - 1];
  }
  if (current) setActiveSection(current.id);
  scrollTicking = false;
}

window.addEventListener('scroll', () => {
  if (!scrollTicking) {
    requestAnimationFrame(syncNavigation);
    scrollTicking = true;
  }
}, { passive: true });
window.addEventListener('resize', syncNavigation);

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      const original = button.textContent;
      button.textContent = 'Copied ✓';
      setTimeout(() => { button.textContent = original; }, 1500);
    } catch {
      button.textContent = 'Select command below';
    }
  });
});

function formatSpeed(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(2)}M chars/s`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(0)}K chars/s`;
  return `${Math.round(number)} chars/s`;
}

function renderLeaderboard(entries) {
  const body = document.querySelector('#leaderboard-body');
  body.replaceChildren();
  entries.forEach((entry) => {
    const row = document.createElement('tr');
    if (entry.status === 'baseline') row.className = 'baseline-row';
    const rank = entry.rank === '—' ? '—' : entry.rank === 1 ? '♛ 1' : String(entry.rank);
    const safeTeam = document.createTextNode(entry.team);
    const teamCell = document.createElement('td');
    teamCell.append(safeTeam);
    if (entry.status === 'baseline') {
      const badge = document.createElement('span');
      badge.className = 'baseline-badge';
      badge.textContent = 'baseline';
      teamCell.append(badge);
    }
    const values = [
      rank,
      teamCell,
      Number(entry.score).toFixed(4),
      formatSpeed(entry.speed_chars_per_second),
      Number(entry.vocab_size).toLocaleString('en'),
    ];
    values.forEach((value) => {
      if (value instanceof HTMLElement) row.append(value);
      else {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.append(cell);
      }
    });
    body.append(row);
  });
}

async function loadLeaderboard() {
  try {
    const response = await fetch('data/leaderboard.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    renderLeaderboard(payload.entries || []);
    const updated = document.querySelector('#board-updated');
    updated.textContent = payload.updated_at
      ? `Updated ${new Intl.DateTimeFormat('en', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(payload.updated_at))}`
      : 'Awaiting first evaluation';
  } catch (error) {
    document.querySelector('#leaderboard-error').hidden = false;
    document.querySelector('#board-updated').textContent = 'Repository results are authoritative';
    document.querySelector('#leaderboard-body').innerHTML = '<tr><td colspan="5" class="loading-cell">Leaderboard unavailable</td></tr>';
  }
}

function hydrateRepositoryLinks() {
  if (!location.hostname.endsWith('github.io')) return;
  const owner = location.hostname.split('.')[0];
  const repository = location.pathname.split('/').filter(Boolean)[0];
  if (!owner || !repository) return;
  document.querySelectorAll('.repo-link').forEach((link) => {
    const path = link.dataset.repoPath;
    link.href = path
      ? `https://github.com/${owner}/${repository}/blob/main/${path}`
      : `https://github.com/${owner}/${repository}`;
  });
}

hydrateRepositoryLinks();
loadLeaderboard();
syncNavigation();

