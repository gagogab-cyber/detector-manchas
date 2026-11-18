(() => {
  const $ = id => document.getElementById(id);
  const SLIDERS = ["tol", "z_thr", "min_area", "crop_izq", "crop_der", "crop_arr", "crop_aba"];
  const CHECKS = { boost: "boost", sombras: "sombras", estampados: "estampados", piel: "piel", paleta: "paleta", fondo: "fondo", cnn: "cnn" };
  let archivo = null;
  let ultimo = null;
  let controlador = null;

  // ---- etiquetas en vivo de los sliders
  SLIDERS.forEach(id => {
    const el = $(id), out = $("o_" + id);
    const pintar = () => { out.textContent = el.value; };
    el.addEventListener("input", pintar);
    pintar();
  });

  // ---- ¿hay CNN entrenada en el servidor?
  fetch("/api/config").then(r => r.json()).then(c => { if (c.cnn) $("cnnWrap").classList.remove("d-none"); }).catch(() => {});

  // ---- carga de archivo: clic, arrastrar y soltar, cámara
  const dz = $("dropzone");
  dz.addEventListener("click", () => $("archivo").click());
  dz.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("archivo").click(); } });
  $("btnCamara").addEventListener("click", () => $("camara").click());
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", e => elegir(e.dataTransfer.files[0]));
  $("archivo").addEventListener("change", e => elegir(e.target.files[0]));
  $("camara").addEventListener("change", e => elegir(e.target.files[0]));

  function elegir(f) {
    if (!f) return;
    if (!/^image\/(jpeg|png)$/.test(f.type) && !/\.(jpe?g|png)$/i.test(f.name)) {
      return mostrarError("Formato no válido. Usa una imagen JPG o PNG.");
    }
    archivo = f;
    $("nombreArchivo").textContent = "📄 " + f.name + " (" + (f.size / 1024).toFixed(0) + " KB)";
    $("btnAnalizar").disabled = false;
    analizar();
  }

  // ---- análisis
  $("btnAnalizar").addEventListener("click", analizar);

  async function analizar() {
    if (!archivo) return;
    if (controlador) controlador.abort();
    controlador = new AbortController();
    const fd = new FormData();
    fd.append("imagen", archivo);
    SLIDERS.forEach(id => fd.append(id, $(id).value));
    fd.append("max_side", $("max_side").value);
    Object.entries(CHECKS).forEach(([k, id]) => fd.append(k, $(id).checked ? "1" : "0"));

    estado("cargando");
    $("btnAnalizar").disabled = true;
    try {
      const r = await fetch("/api/analizar", { method: "POST", body: fd, signal: controlador.signal });
      const data = await r.json().catch(() => ({ error: "Respuesta inválida del servidor." }));
      if (!r.ok) throw new Error(data.error || "Error al analizar la imagen.");
      mostrar(data);
    } catch (e) {
      if (e.name === "AbortError") return;
      mostrarError(e.message);
    } finally {
      $("btnAnalizar").disabled = false;
    }
  }

  function estado(cual) {
    $("vacio").classList.toggle("d-none", cual !== "vacio");
    $("cargando").classList.toggle("d-none", cual !== "cargando");
    $("resultados").classList.toggle("d-none", cual !== "resultados");
    $("error").classList.add("d-none");
  }

  function mostrarError(msg) {
    estado(ultimo ? "resultados" : "vacio");
    const el = $("error");
    el.textContent = msg;
    el.classList.remove("d-none");
  }

  function mostrar(d) {
    ultimo = d;
    estado("resultados");
    const v = $("veredicto");
    v.textContent = d.total ? "✖ " + d.veredicto : "✔ " + d.veredicto;
    v.className = "verdict " + (d.total ? "bad" : "ok");
    $("tagFondo").textContent = d.fondo_separado ? "Prenda separada del fondo ✓ (el % de área dañada es sobre la prenda)" : "";
    $("m_total").textContent = d.total;
    $("m_area").textContent = d.area_danada_pct + "%";
    $("m_tiempo").textContent = d.tiempo_ms;
    $("img_original").src = d.original;
    $("img_marcada").src = d.marcada;
    $("img_heatmap").src = d.heatmap;
    $("dlPng").href = d.marcada;
    $("tablaWrap").classList.toggle("d-none", !d.total);

    const tb = $("tbody");
    tb.replaceChildren();
    d.defectos.forEach((x, i) => {
      const tr = document.createElement("tr");
      [i + 1, x.x, x.y, x.ancho, x.alto, x["area_%"], x["ΔE medio"], x["ΔE máx"], x.tipo, x.nivel].forEach((val, j) => {
        const td = document.createElement("td");
        td.textContent = val;
        if (j === 9) td.className = x.nivel === "grave" ? "text-danger fw-bold" : "text-warning";
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  }

  // ---- CSV
  $("dlCsv").addEventListener("click", () => {
    if (!ultimo) return;
    const cols = ["x", "y", "ancho", "alto", "area_%", "ΔE medio", "ΔE máx", "tipo", "nivel"];
    const filas = [cols.join(",")].concat(ultimo.defectos.map(d => cols.map(c => d[c]).join(",")));
    const blob = new Blob(["\ufeff" + filas.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "defectos.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  // ---- ampliar imágenes
  const lb = $("lightbox");
  document.querySelectorAll(".shot img").forEach(img => img.addEventListener("click", () => {
    lb.querySelector("img").src = img.src;
    lb.classList.remove("d-none");
  }));
  lb.addEventListener("click", () => lb.classList.add("d-none"));
  document.addEventListener("keydown", e => { if (e.key === "Escape") lb.classList.add("d-none"); });
})();
