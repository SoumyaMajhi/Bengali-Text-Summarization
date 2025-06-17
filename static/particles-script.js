let currentParticleMode = localStorage.getItem('theme') || 'light';

const lightParticles = {
  particles: {
    number: { value: 80, density: { enable: true, value_area: 800 } },
    color: { value: '#040434' },
    shape: { type: 'circle' },
    opacity: { value: 0.5 },
    size: { value: 3, random: true },
    line_linked: {
      enable: true,
      distance: 150,
      color: '#040434',
      opacity: 0.4,
      width: 1
    },
    move: { enable: true, speed: 6, direction: 'none' }
  },
  interactivity: {
    detect_on: 'canvas',
    events: {
      onhover: { enable: true, mode: 'repulse' },
      onclick: { enable: true, mode: 'push' },
      resize: true
    }
  },
  retina_detect: true
};

const darkParticles = JSON.parse(JSON.stringify(lightParticles));
darkParticles.particles.color.value = '#ffffff';
darkParticles.particles.line_linked.color = '#ffffff';

function loadParticles(theme = 'light') {
  document.getElementById('particles-js').innerHTML = '';
  particlesJS('particles-js', theme === 'dark' ? darkParticles : lightParticles);
}

document.addEventListener("DOMContentLoaded", () => {
  loadParticles(currentParticleMode);

  // Watch for theme changes
  const toggle = document.querySelector('.theme-switch__checkbox');
  if (toggle) {
    toggle.addEventListener('change', () => {
      const newTheme = document.body.classList.contains('dark-mode') ? 'dark' : 'light';
      loadParticles(newTheme);
    });
  }
});
