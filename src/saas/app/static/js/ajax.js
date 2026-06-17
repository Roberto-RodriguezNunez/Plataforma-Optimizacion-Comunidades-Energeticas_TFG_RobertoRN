/**
 * LeaLink — Progressive Enhancement AJAX
 * Intercepta formularios con data-ajax-action y los envía sin recargar página.
 * Con JS desactivado, el formulario funciona normalmente (POST → redirect).
 */
document.addEventListener('DOMContentLoaded', () => {
  const csrfMeta = document.querySelector('meta[name="csrf-token"]');
  const csrfToken = csrfMeta ? csrfMeta.getAttribute('content') : '';

  // Interceptar todos los formularios marcados con data-ajax-action
  document.querySelectorAll('form[data-ajax-action]').forEach(form => {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const url = form.dataset.ajaxAction;
      const formData = new FormData(form);

      // Asegurar que el token CSRF va en los datos
      if (!formData.has('csrf_token') && csrfToken) {
        formData.set('csrf_token', csrfToken);
      }

      try {
        const resp = await fetch(url, {
          method: 'POST',
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
          body: formData
        });

        let data;
        try {
          data = await resp.json();
        } catch {
          mostrarFlash('Respuesta inesperada del servidor.', 'danger');
          return;
        }

        if (data.success) {
          // Si es "marcar leída": actualizar visualmente sin eliminar el card
          if (form.dataset.ajaxNotif === 'decrement') {
            const card = form.closest('[data-removable]');
            if (card) {
              card.classList.remove('notif-no-leida');
              card.querySelectorAll('[data-notif-dot]').forEach(el => el.remove());
              card.querySelectorAll('[data-notif-btn-leer]').forEach(el => el.remove());
            }
          } else {
            // Eliminar el elemento más cercano con data-removable
            const removable = form.closest('[data-removable]');
            if (removable) removable.remove();
          }

          if (data.message) mostrarFlash(data.message, 'success');

          // Actualizar contador de notificaciones en campana si aplica
          if (form.dataset.ajaxNotif) actualizarCampana(form.dataset.ajaxNotif);

          // Marcar todas leídas: recargar la lista limpiando visualmente
          if (form.dataset.ajaxMarkAll) {
            document.querySelectorAll('.notif-no-leida').forEach(el => el.classList.remove('notif-no-leida'));
            document.querySelectorAll('[data-notif-dot]').forEach(el => el.remove());
            document.querySelectorAll('[data-notif-btn-leer]').forEach(el => el.remove());
            actualizarCampana('reset');
          }

          // Redirigir si el servidor lo indica
          if (data.redirect) {
            window.location.href = data.redirect;
          }
        } else {
          mostrarFlash(data.error || 'Ha ocurrido un error.', 'danger');
        }
      } catch (err) {
        mostrarFlash('Error de conexión. Inténtalo de nuevo.', 'danger');
      }
    });
  });

  function actualizarCampana(modo) {
    const badge = document.querySelector('[data-notif-count]');
    if (!badge) return;
    if (modo === 'reset') {
      badge.textContent = '0';
      badge.style.display = 'none';
      return;
    }
    const actual = parseInt(badge.textContent || '0', 10);
    const nuevo = modo === 'decrement' ? Math.max(0, actual - 1) : actual;
    badge.textContent = String(nuevo);
    badge.style.display = nuevo > 0 ? '' : 'none';
  }

  function mostrarFlash(mensaje, categoria) {
    let contenedor = document.querySelector('.flash-container');
    if (!contenedor) {
      contenedor = document.createElement('div');
      contenedor.className = 'flash-container';
      contenedor.setAttribute('aria-live', 'polite');
      const main = document.querySelector('main');
      if (main) main.prepend(contenedor);
    }

    const div = document.createElement('div');
    div.className = `flash flash-${categoria}`;
    div.setAttribute('role', 'alert');
    div.textContent = mensaje;
    contenedor.appendChild(div);

    // Auto-eliminar tras 5 segundos
    setTimeout(() => div.remove(), 5000);
  }
});
