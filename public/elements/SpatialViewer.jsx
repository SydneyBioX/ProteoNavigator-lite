import React, { useEffect, useMemo, useRef, useState } from "react";

const UNKNOWN = new Set(["unknown", "unassigned", "uncertain", "nan", "none"]);

function categoryColor(label) {
  if (UNKNOWN.has(String(label).toLowerCase())) return "#9ca3af";
  let hash = 0;
  for (const char of String(label)) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return `hsl(${hash % 360} 68% 53%)`;
}

const VIRIDIS_FAMILY = {
  viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
  plasma: ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
  magma: ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"],
  cividis: ["#00224e", "#26456e", "#576d8c", "#8a8d8c", "#c3ad72", "#fee838"],
};

function interpolateHex(left, right, amount) {
  const channel = (hex, offset) => parseInt(hex.slice(offset, offset + 2), 16);
  const value = (offset) => Math.round(
    channel(left, offset) + (channel(right, offset) - channel(left, offset)) * amount
  ).toString(16).padStart(2, "0");
  return `#${value(1)}${value(3)}${value(5)}`;
}

function paletteColor(paletteName, value) {
  const colors = VIRIDIS_FAMILY[paletteName] || VIRIDIS_FAMILY.viridis;
  const position = Math.max(0, Math.min(1, value)) * (colors.length - 1);
  const left = Math.floor(position);
  const right = Math.min(colors.length - 1, left + 1);
  return interpolateHex(colors[left], colors[right], position - left);
}

function uncertaintyColor(value, paletteName, minimum, maximum) {
  if (value == null || !Number.isFinite(Number(value))) return "#9ca3af";
  const span = maximum - minimum;
  const t = span > 0 ? (Number(value) - minimum) / span : 0.5;
  return paletteColor(paletteName, t);
}

export default function SpatialViewer() {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const hasLabels = Array.isArray(props.labels);
  const markerNames = Array.isArray(props.markerNames) ? props.markerNames : [];
  const hasMarkers = markerNames.length > 0 && Array.isArray(props.markerValues);
  const uncertaintyNames = props.uncertaintyMetrics ? Object.keys(props.uncertaintyMetrics) : [];
  const hasUncertainty = uncertaintyNames.length > 0;
  const [mode, setMode] = useState(hasLabels ? "labels" : "coordinates");
  const [marker, setMarker] = useState(markerNames[0] || null);
  const [uncertaintyMetric, setUncertaintyMetric] = useState(
    uncertaintyNames.includes("Normalized entropy") ? "Normalized entropy" : uncertaintyNames[0]
  );
  const sampleOptions = useMemo(
    () => Array.isArray(props.samples) ? [...new Set(props.samples)].sort() : [],
    [props.samples]
  );
  const [sample, setSample] = useState(sampleOptions[0] || null);

  useEffect(() => {
    setMode(hasLabels ? "labels" : "coordinates");
    setSample(sampleOptions[0] || null);
    setMarker(markerNames[0] || null);
    setUncertaintyMetric(
      uncertaintyNames.includes("Normalized entropy") ? "Normalized entropy" : uncertaintyNames[0]
    );
  }, [props.title, hasLabels]);

  const indices = useMemo(() => {
    const all = props.x.map((_, index) => index);
    if (!sample || !props.samples) return all;
    return all.filter((index) => props.samples[index] === sample);
  }, [props.x, props.samples, sample]);

  const markerIndex = markerNames.indexOf(marker);
  const markerRange = useMemo(() => {
    if (!hasMarkers || markerIndex < 0) return {minimum: 0, maximum: 1};
    const values = indices
      .map((index) => Number(props.markerValues[markerIndex][index]))
      .filter((value) => Number.isFinite(value));
    return values.length
      ? {minimum: Math.min(...values), maximum: Math.max(...values)}
      : {minimum: 0, maximum: 1};
  }, [hasMarkers, indices, markerIndex, props.markerValues]);

  const categories = useMemo(() => {
    if (!hasLabels) return [];
    const counts = new Map();
    for (const index of indices) {
      const label = props.labels[index] || "Unassigned";
      counts.set(label, (counts.get(label) || 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [hasLabels, indices, props.labels]);
  const selectedTotal = sample && props.sampleCounts
    ? props.sampleCounts[sample]
    : props.totalCells;
  const uncertaintyPalette = props.uncertaintyPalettes?.[uncertaintyMetric] || "viridis";
  const uncertaintyRange = useMemo(() => {
    if (!hasUncertainty || !uncertaintyMetric) return {minimum: 0, maximum: 1};
    const values = indices
      .map((index) => Number(props.uncertaintyMetrics[uncertaintyMetric][index]))
      .filter((value) => Number.isFinite(value));
    return values.length
      ? {minimum: Math.min(...values), maximum: Math.max(...values)}
      : {minimum: 0, maximum: 1};
  }, [hasUncertainty, indices, uncertaintyMetric, props.uncertaintyMetrics]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !indices.length) return;

    const draw = () => {
      const width = Math.max(320, container.clientWidth);
      const height = Math.max(420, Math.min(720, window.innerHeight - 220));
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const context = canvas.getContext("2d");
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);

      const xs = indices.map((index) => Number(props.x[index]));
      const ys = indices.map((index) => Number(props.y[index]));
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const padding = 16;
      const scale = Math.min(
        (width - padding * 2) / Math.max(maxX - minX, 1),
        (height - padding * 2) / Math.max(maxY - minY, 1)
      );
      const offsetX = (width - (maxX - minX) * scale) / 2;
      const offsetY = (height - (maxY - minY) * scale) / 2;
      const radius = indices.length > 20000 ? 1 : indices.length > 5000 ? 1.35 : 2;

      context.globalAlpha = indices.length > 15000 ? 0.72 : 0.88;
      for (const index of indices) {
        const x = offsetX + (Number(props.x[index]) - minX) * scale;
        const y = offsetY + (Number(props.y[index]) - minY) * scale;
        context.fillStyle = mode === "labels"
          ? categoryColor(props.labels[index] || "Unassigned")
          : mode === "uncertainty"
            ? uncertaintyColor(
                props.uncertaintyMetrics[uncertaintyMetric][index],
                uncertaintyPalette,
                uncertaintyRange.minimum,
                uncertaintyRange.maximum
              )
            : hasMarkers && markerIndex >= 0
              ? uncertaintyColor(
                  props.markerValues[markerIndex][index],
                  "viridis",
                  markerRange.minimum,
                  markerRange.maximum
                )
              : "#38bdf8";
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fill();
      }
      context.globalAlpha = 1;
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(container);
    return () => observer.disconnect();
  }, [indices, mode, markerIndex, markerRange, uncertaintyMetric, uncertaintyPalette,
      uncertaintyRange, props.x, props.y, props.labels, props.markerValues,
      props.uncertaintyMetrics]);

  return (
    <div className="rounded-xl border bg-background overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b p-4">
        <div>
          <h3 className="font-semibold">{props.title}</h3>
          <p className="text-xs text-muted-foreground mt-1">
            {sample ? `${sample} · ` : ""}Showing {indices.length.toLocaleString()} points · {selectedTotal.toLocaleString()} cells
            {indices.length < selectedTotal ? " · deterministically downsampled" : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {sampleOptions.length > 1 && (
            <select className="rounded-md border bg-background px-2 py-1 text-sm" value={sample}
              onChange={(event) => setSample(event.target.value)}>
              {sampleOptions.map((value) => <option key={value}>{value}</option>)}
            </select>
          )}
          <div className="inline-flex rounded-md border p-0.5 text-sm">
            <button className={`rounded px-3 py-1 ${mode === "coordinates" ? "bg-muted font-medium" : ""}`}
              onClick={() => setMode("coordinates")}>Input</button>
            {hasLabels && <button className={`rounded px-3 py-1 ${mode === "labels" ? "bg-muted font-medium" : ""}`}
              onClick={() => setMode("labels")}>Cell type</button>}
            {hasUncertainty && <button className={`rounded px-3 py-1 ${mode === "uncertainty" ? "bg-muted font-medium" : ""}`}
              onClick={() => setMode("uncertainty")}>Uncertainty</button>}
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-[minmax(0,1fr)_220px]">
        <div ref={containerRef} className="min-w-0 bg-slate-950/95">
          <canvas ref={canvasRef} className="block" />
        </div>
        <aside className="border-l p-3 max-h-[520px] overflow-auto text-xs">
          {mode === "labels" && categories.map(([label, count]) => (
            <div key={label} className="flex items-center justify-between gap-2 py-1">
              <span className="flex min-w-0 items-center gap-2">
                <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{background: categoryColor(label)}} />
                <span className="truncate" title={label}>{label}</span>
              </span>
              <span className="tabular-nums text-muted-foreground">{count.toLocaleString()}</span>
            </div>
          ))}
          {mode === "uncertainty" && (
            <div className="space-y-3">
              <select className="w-full rounded-md border bg-background px-2 py-1.5"
                value={uncertaintyMetric} onChange={(event) => setUncertaintyMetric(event.target.value)}>
                {uncertaintyNames.map((name) => <option key={name}>{name}</option>)}
              </select>
              <div className="h-3 rounded-full" style={{
                background: `linear-gradient(90deg,${(VIRIDIS_FAMILY[uncertaintyPalette] || VIRIDIS_FAMILY.viridis).join(",")})`
              }} />
              <div className="flex justify-between tabular-nums">
                <span>{uncertaintyRange.minimum.toFixed(3)} · lower</span>
                <span>{uncertaintyRange.maximum.toFixed(3)} · higher</span>
              </div>
              <p className="text-muted-foreground">
                {uncertaintyPalette} palette · scaled independently for this metric and sample
              </p>
              <p className="leading-relaxed text-muted-foreground">{props.uncertaintyNote}</p>
            </div>
          )}
          {mode === "coordinates" && (
            <div className="space-y-3">
              {hasMarkers ? (
                <>
                  <select className="w-full rounded-md border bg-background px-2 py-1.5"
                    value={marker || ""} onChange={(event) => setMarker(event.target.value)}>
                    {markerNames.map((name) => <option key={name}>{name}</option>)}
                  </select>
                  <div className="h-3 rounded-full" style={{
                    background: `linear-gradient(90deg,${VIRIDIS_FAMILY.viridis.join(",")})`
                  }} />
                  <div className="flex justify-between tabular-nums">
                    <span>{markerRange.minimum.toFixed(3)} · lower</span>
                    <span>{markerRange.maximum.toFixed(3)} · higher</span>
                  </div>
                  <p className="leading-relaxed text-muted-foreground">
                    {marker} intensity from <code>{props.markerAssay}</code>. Viridis is scaled
                    independently for the selected sample.
                  </p>
                  {props.markersTruncated && (
                    <p className="text-muted-foreground">Showing the first 200 markers.</p>
                  )}
                </>
              ) : (
                <p className="leading-relaxed text-muted-foreground">
                  Raw spatial coordinates from <code>obsm["spatial"]</code>. Each point is one cell.
                </p>
              )}
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
