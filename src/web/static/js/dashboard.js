/* DDOS 仪表盘图表 — 读取 window.DASHBOARD_DATA 渲染三张 ECharts */
(function () {
  "use strict";

  if (typeof echarts === "undefined") {
    return; // ECharts 未加载，静默跳过（页面表格仍可用）
  }

  var data = window.DASHBOARD_DATA || {};
  var charts = [
    { id: "chart-category", title: "分类分布（大类）", kind: "pie", rows: data.category || [] },
    { id: "chart-industry", title: "行业分布（行业域）", kind: "bar", rows: data.industry || [] },
    { id: "chart-direction", title: "评分方向分布", kind: "bar", rows: data.direction || [] }
  ];

  charts.forEach(function (c) {
    var el = document.getElementById(c.id);
    if (!el) return;

    var nonEmpty = (c.rows || []).some(function (r) { return r.value > 0; });
    if (!nonEmpty) {
      el.innerHTML = '<div class="empty" style="text-align:center;padding-top:110px;">暂无数据</div>';
      return;
    }

    var chart = echarts.init(el);
    var option = {
      title: { text: c.title, left: 12, top: 6, textStyle: { fontSize: 13 } },
      tooltip: { trigger: c.kind === "pie" ? "item" : "axis" },
      grid: c.kind === "bar" ? { left: 40, right: 16, top: 44, bottom: 40 } : undefined,
      series: []
    };

    if (c.kind === "pie") {
      option.color = ["#2b5ce6", "#16a34a", "#dc2626", "#f59e0b", "#8b5cf6", "#06b6d4", "#64748b"];
      option.series = [{
        type: "pie",
        radius: ["35%", "68%"],
        center: ["50%", "58%"],
        avoidLabelOverlap: true,
        itemStyle: { borderRadius: 4, borderColor: "#fff", borderWidth: 2 },
        label: { formatter: "{b}: {c}" },
        data: c.rows
      }];
    } else {
      option.series = [{
        type: "bar",
        data: c.rows.map(function (r) { return r.value; }),
        barMaxWidth: 46,
        itemStyle: { color: "#2b5ce6", borderRadius: [4, 4, 0, 0] }
      }];
      option.xAxis = { type: "category", data: c.rows.map(function (r) { return r.name; }), axisLabel: { interval: 0, rotate: c.rows.length > 4 ? 20 : 0 } };
      option.yAxis = { type: "value", minInterval: 1 };
    }

    chart.setOption(option);
    window.addEventListener("resize", function () { chart.resize(); });
  });
})();
