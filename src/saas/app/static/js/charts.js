/**
 * LeaLink — Chart.js 4
 * Paleta light theme (blanco/negro/gris — Trade Republic style)
 */
document.addEventListener('DOMContentLoaded', () => {

  const C = {
    // Colores principales
    green:      '#00C896',
    greenBg:    'rgba(0,200,150,.12)',
    red:        '#E53E3E',
    redBg:      'rgba(229,62,62,.1)',
    blue:       '#3B82F6',
    blueBg:     'rgba(59,130,246,.1)',
    amber:      '#D97706',
    amberBg:    'rgba(217,119,6,.1)',
    grey:       '#aaa',
    greyBg:     'rgba(0,0,0,.06)',
    // Ejes y cuadrícula (tema claro)
    text:       '#999',
    grid:       'rgba(0,0,0,.07)',
    // Comunidad — vivienda actual vs resto
    highlight:  '#00C896',
    highlightBg:'rgba(0,200,150,.75)',
    other:      'rgba(0,0,0,.1)',
    otherLine:  'rgba(0,0,0,.2)',
    // Doughnut — origen energía
    d0: '#00C896',   // autoconsumo directo (sol propio)
    d0b: '#6EE7B7',  // autoconsumo indirecto (sol compartido de la comunidad)
    d1: '#3B82F6',   // batería comunitaria
    d2: '#E53E3E',   // red eléctrica
    d3: '#aaa',      // residual
  };

  // Opciones de ejes — tema claro
  function scalesXY(stacked = false) {
    return {
      x: {
        stacked,
        ticks:  { color: C.text, font: { size: 11 } },
        grid:   { color: C.grid },
        border: { color: C.grid },
      },
      y: {
        stacked,
        beginAtZero: true,
        ticks:  { color: C.text, font: { size: 11 } },
        grid:   { color: C.grid },
        border: { color: C.grid },
      },
    };
  }

  function legendOpts(position = 'bottom') {
    return {
      position,
      labels: {
        color: C.text,
        font: { size: 11 },
        padding: 16,
        boxWidth: 10,
        boxHeight: 10,
      },
    };
  }

  function tooltipOpts() {
    return {
      backgroundColor: '#fff',
      titleColor: '#0a0a0a',
      bodyColor: '#666',
      borderColor: '#e5e5e5',
      borderWidth: 1,
      padding: 10,
      cornerRadius: 8,
    };
  }

  // ---------------------------------------------------------------
  const renderers = {

    /** Línea — ahorro con comunidad vs los 2 escenarios base */
    'line-ahorro': (canvas, datos) => {
      new Chart(canvas, {
        type: 'line',
        data: {
          labels: datos.labels,
          datasets: [
            {
              label: 'vs Sin paneles',
              data: datos.vs_sin_paneles,
              borderColor: C.red,
              backgroundColor: 'rgba(229,62,62,.07)',
              borderWidth: 2,
              tension: 0.35,
              fill: false,
              pointBackgroundColor: C.red,
              pointRadius: 3,
              pointHoverRadius: 5,
            },
            {
              label: 'vs Sin comunidad',
              data: datos.vs_solo_paneles,
              borderColor: C.amber,
              backgroundColor: C.amberBg,
              borderWidth: 2,
              tension: 0.35,
              fill: false,
              pointBackgroundColor: C.amber,
              pointRadius: 3,
              pointHoverRadius: 5,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            legend: legendOpts('top'),
            tooltip: {
              ...tooltipOpts(),
              callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} €` },
            },
          },
          scales: scalesXY(),
        },
      });
    },

    /** Barras agrupadas — 3 escenarios */
    'bar-compare': (canvas, datos) => {
      new Chart(canvas, {
        type: 'bar',
        data: {
          labels: datos.labels,
          datasets: [
            {
              label: 'Sin paneles',
              data: datos.sin_paneles,
              backgroundColor: 'rgba(229,62,62,.18)',
              borderColor: C.red,
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: 'Sin comunidad',
              data: datos.solo_paneles,
              backgroundColor: C.amberBg,
              borderColor: C.amber,
              borderWidth: 1,
              borderRadius: 4,
            },
            {
              label: 'Con comunidad',
              data: datos.comunidad_completa,
              backgroundColor: C.greenBg,
              borderColor: C.green,
              borderWidth: 1,
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: legendOpts('top'),
            tooltip: {
              ...tooltipOpts(),
              callbacks: {
                label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} €`,
              },
            },
          },
          scales: scalesXY(),
        },
      });
    },

    /** Doughnut — mix energético (minimalista) */
    'doughnut-mix': (canvas, datos) => {
      const labels = datos.labels;
      // Orden de segmentos: autoconsumo directo → indirecto → batería → red
      const palette = [C.d0, C.d0b, C.d1, C.d2];
      new Chart(canvas, {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{
            data: datos.datos,
            backgroundColor: palette.slice(0, labels.length),
            borderColor: '#fff',
            borderWidth: 2,
            hoverOffset: 4,
          }],
        },
        options: {
          responsive: true,
          cutout: '70%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                color: '#666',
                font: { size: 11 },
                padding: 14,
                boxWidth: 10,
                boxHeight: 10,
                usePointStyle: true,
                pointStyleWidth: 10,
              },
            },
            tooltip: {
              ...tooltipOpts(),
              callbacks: {
                label: ctx => {
                  const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                  const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : 0;
                  return ` ${ctx.parsed.toFixed(1)} kWh — ${pct}%`;
                },
              },
            },
          },
        },
      });
    },

    /** Barras horizontales — comparativa ahorro medio en la comunidad */
    'bar-community': (canvas, datos) => {
      const hi = datos.highlight;
      const n  = datos.labels.length;

      const bgColors = datos.labels.map((_, i) =>
        i === hi ? C.highlightBg : C.other
      );
      const borderColors = datos.labels.map((_, i) =>
        i === hi ? C.green : C.otherLine
      );
      new Chart(canvas, {
        type: 'bar',
        data: {
          labels: datos.labels,
          datasets: [{
            label: 'Ahorro medio mensual (€)',
            data: datos.medias,
            backgroundColor: bgColors,
            borderColor: borderColors,
            borderWidth: 1,
            borderRadius: 3,
          }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              ...tooltipOpts(),
              callbacks: {
                label: ctx => ` ${ctx.parsed.x.toFixed(2)} €/mes`,
                title: ctxs => {
                  const idx = ctxs[0].dataIndex;
                  return idx === hi
                    ? `${ctxs[0].label}  ← Esta vivienda`
                    : ctxs[0].label;
                },
              },
            },
          },
          scales: {
            x: {
              beginAtZero: true,
              ticks:  { color: C.text, font: { size: 11 } },
              grid:   { color: C.grid },
              border: { color: C.grid },
            },
            y: {
              ticks: {
                color:  ctx => ctx.index === hi ? C.green : C.text,
                font:   ctx => ({ size: 11, weight: ctx.index === hi ? '700' : '400' }),
                autoSkip: false,
              },
              grid: { display: false },
            },
          },
        },
      });
    },

    /** Línea de área — ahorro comunidad (detalle comunidad) */
    'line-ahorro-com': (canvas, datos) => {
      new Chart(canvas, {
        type: 'line',
        data: {
          labels: datos.labels,
          datasets: [{
            label: 'Ahorro comunidad (€)',
            data: datos.ahorro,
            borderColor: C.black,
            backgroundColor: 'rgba(0,0,0,.04)',
            borderWidth: 2,
            tension: 0.35,
            fill: true,
            pointBackgroundColor: C.black,
            pointRadius: 3,
            pointHoverRadius: 5,
          }],
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false },
            tooltip: {
              ...tooltipOpts(),
              callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(2)} €` },
            },
          },
          scales: scalesXY(),
        },
      });
    },

  };

  // Escanear y renderizar
  document.querySelectorAll('canvas[data-chart-type]').forEach(canvas => {
    const type = canvas.dataset.chartType;
    const raw  = canvas.dataset.chart;
    if (!raw || !renderers[type]) return;
    try {
      renderers[type](canvas, JSON.parse(raw));
    } catch (e) {
      console.warn(`[charts.js] Error en ${type}:`, e);
    }
  });

});
