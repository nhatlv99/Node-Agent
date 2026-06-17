// Verify _chartOption builds a valid ECharts option for every chart type,
// WITHOUT a browser. We slice ONLY the chart-helper function declarations out
// of index.html (not the whole app body, which references browser globals like
// localStorage) and eval just those with a minimal echarts stub.
import fs from "node:fs";
import path from "node:path";

const HTML = path.resolve("node_agent/static/index.html");
const html = fs.readFileSync(HTML, "utf8");

// Grab a top-level `function NAME(...) { ... }` block by brace-matching.
function sliceFn(src, name){
  const start = src.indexOf("function " + name);
  if (start < 0) throw new Error("not found: " + name);
  let i = src.indexOf("{", start), depth = 0;
  for (let j = i; j < src.length; j++){
    if (src[j] === "{") depth++;
    else if (src[j] === "}"){ depth--; if (depth === 0) return src.slice(start, j+1); }
  }
  throw new Error("unbalanced: " + name);
}

const NEED = ["_rgba","_grad","_esc","_chartHead","_gnChartTheme","_axisCat","_axisVal","_chartOption","_chartOptionRaw"];
const fns = NEED.map(n => sliceFn(html, n)).join("\n\n");
// _GN_GREENS constant (palette) is referenced by the helpers.
const palette = html.match(/const _GN_GREENS = \[[^\]]*\];/)[0];

const echarts = {
  graphic: { LinearGradient: function (x0,y0,x1,y1,stops){ this.type="linearGradient"; this.colorStops=stops; } },
};

const factory = new Function("echarts", `
  ${palette}
  ${fns}
  return { _chartOption, _chartHead };
`);
const { _chartOption, _chartHead } = factory(echarts);

const cases = [
  ["bar",         {type:"bar",   title:"VRAM", unit:"GB", labels:["H100","H200"], series:[{name:"Dung lượng",data:[80,141]}]}],
  ["bar-multi",   {type:"bar",   title:"So sánh", unit:"", labels:["H100","H200"], series:[{name:"VRAM",data:[80,141]},{name:"TF",data:[1979,2200]}]}],
  ["hbar",        {type:"hbar",  title:"Giá", unit:"USD/h", labels:["Basic","Standard","Premium"], series:[{name:"Giá",data:[0.39,2.69,2.99]}]}],
  ["line",        {type:"line",  title:"Giá theo tháng", unit:"USD", labels:["T1","T2","T3"], series:[{name:"H100",data:[3.0,2.9,2.69]}]}],
  ["area",        {type:"area",  title:"Lưu lượng", unit:"GB", labels:["T1","T2","T3"], series:[{name:"Egress",data:[12,18,25]}]}],
  ["pie",         {type:"pie",   title:"Thị phần", unit:"%", labels:["A","B","C"], series:[{name:"Share",data:[50,30,20]}]}],
  ["donut",       {type:"donut", title:"Cơ cấu chi phí", unit:"%", labels:["Compute","Storage","Network"], series:[{name:"Tỉ trọng",data:[70,20,10]}]}],
  ["radar",       {type:"radar", title:"Hồ sơ", labels:["VRAM","BW","FP16","Giá"], series:[{name:"H100",data:[80,3350,1979,3]},{name:"H200",data:[141,4800,2200,4]}]}],
  ["gauge",       {type:"gauge", title:"Uptime", unit:"%", value:99.95, max:100}],
  ["scatter",     {type:"scatter", title:"Giá vs hiệu năng", labels:[], series:[{name:"GPU",data:[[3,1979],[4,2200]]}]}],
  ["candlestick", {type:"candlestick", title:"VN30", unit:"đ", labels:["T2","T3","T4"], series:[{name:"OHLC",data:[[10,12,9,13],[12,11,10,14],[11,13,11,15]]}]}],
  ["histogram",   {type:"histogram", title:"Phân bố latency", unit:"req", labels:["0-50","50-100","100-150","150-200"], series:[{name:"Tần suất",data:[12,40,28,9]}]}],
  ["boxplot",     {type:"boxplot", title:"Phân tán giá", unit:"USD", labels:["H100","H200"], series:[{name:"Giá",data:[[2.5,2.7,2.9,3.1,3.4],[3.8,4.0,4.2,4.5,4.9]]}]}],
  ["heatmap",     {type:"heatmap", title:"Mức dùng GPU", unit:"%", xlabels:["T2","T3","T4"], ylabels:["Sáng","Chiều","Tối"], series:[{name:"Dùng",data:[[0,0,30],[1,0,55],[2,0,80],[0,1,40],[1,1,60],[2,1,75],[0,2,20],[1,2,35],[2,2,90]]}]}],
  ["funnel",      {type:"funnel", title:"Phễu chuyển đổi", unit:"user", labels:["Đăng ký","Dùng thử","Trả phí"], series:[{name:"User",data:[1000,420,130]}]}],
  ["treemap",     {type:"treemap", title:"Tỉ trọng dịch vụ", unit:"%", labels:["Compute","Storage","Network","AI","Backup","Other"], series:[{name:"Share",data:[40,20,15,12,8,5]}]}],
];

let pass=0, fail=0;
for (const [name, spec] of cases){
  try {
    const opt = _chartOption(spec);
    const okType = opt && Array.isArray(opt.series) && opt.series.length>0;
    const seriesType = opt.series[0] && opt.series[0].type;
    if (!okType){ console.log(`x ${name.padEnd(12)} -> no series`); fail++; continue; }
    console.log(`ok ${name.padEnd(12)} -> series.type=${seriesType}`);
    pass++;
  } catch(e){
    console.log(`x ${name.padEnd(12)} -> ${e.message}`); fail++;
  }
}

// framed=true must strip ECharts' own title + legend (header drawn in HTML).
try {
  const fspec = {type:"bar", framed:true, title:"VRAM", unit:"GB", labels:["H100","H200"], series:[{name:"VRAM",data:[80,141]},{name:"BW",data:[3,4]}]};
  const opt = _chartOption(fspec);
  if (opt.title===null && opt.legend===undefined){ console.log("ok framed-strip  -> title/legend removed"); pass++; }
  else { console.log(`x framed-strip  -> title=${opt.title} legend=${JSON.stringify(opt.legend)}`); fail++; }
  // header HTML must contain the title and one legend chip per series.
  const head = _chartHead(fspec);
  if (head.includes("VRAM") && (head.match(/chart-lg-d/g)||[]).length===2){ console.log("ok chart-head    -> title + 2 legend chips"); pass++; }
  else { console.log(`x chart-head    -> ${head.slice(0,120)}`); fail++; }
} catch(e){ console.log(`x framed         -> ${e.message}`); fail++; }
console.log(`\n${pass} pass / ${fail} fail`);
process.exit(fail?1:0);
