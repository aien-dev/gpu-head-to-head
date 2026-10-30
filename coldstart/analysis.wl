(* coldstart/PROTOCOL.md section 7. Evaluated in the Wolfram MCP kernel with results/coldstart/<run>/rows.wl
   (coldstart/run.py) prepended. csRows row: {arm, round, ok, load1, build, ready, firstResult, proc, exec}, ns.
   csWarm row: {arm, round, ok, {10 warm build times, ns}}. Statistics are h2hStats of results/wolfram/analysis.wl,
   in microseconds. A comparison is a WIN or LOSS only when the two IQRs [q1, q3] do not overlap; otherwise
   OVERLAPPING. No significance tests. Output: RawJSON. *)
h2hStats[v_List] := If[Length[v] == 0, <|"n" -> 0|>,
  <|"n" -> Length[v], "median" -> N[Median[v]], "q1" -> N[Quantile[v, 1/4]], "q3" -> N[Quantile[v, 3/4]],
    "iqr" -> N[InterquartileRange[v]], "p50" -> N[Quantile[v, 1/2]], "p99" -> N[Quantile[v, 99/100]]|>];
csArms = {"CUDAPP", "PTXC", "NVRTC"};
csMetrics = {"build" -> 5, "ready" -> 6, "firstResult" -> 7, "proc" -> 8, "exec" -> 9};
(* valid rows: process ok and load1 < 6 at spawn (section 6) *)
csValid[rows_] := Select[rows, #[[3]] === True && #[[4]] < 6 &];
csCall[a_Association, b_Association] := Which[
  a["q3"] < b["q1"], "CUDAPP_LOWER", b["q3"] < a["q1"], "CUDAPP_HIGHER", True, "OVERLAPPING"];
csAnalyse[rows_, warm_] := Module[{valid, per, warmPer, out = <||>, cmp = <||>},
  valid = csValid[rows];
  per = AssociationMap[Function[arm, Module[{rs = Select[valid, #[[1]] == arm &]},
      Join[<|"rows_measured" -> Count[rows, r_ /; r[[1]] == arm],
             "rows_valid" -> Length[rs],
             "rows_failed" -> Count[rows, r_ /; r[[1]] == arm && r[[3]] =!= True],
             "rows_load_excluded" -> Count[rows, r_ /; r[[1]] == arm && r[[3]] === True && r[[4]] >= 6],
             "arm_valid" -> Length[rs] >= 28|>,
        Association[(First[#] -> h2hStats[N[rs[[All, Last[#]]]/1000]]) & /@ csMetrics]]]], csArms];
  warmPer = AssociationMap[Function[arm,
      h2hStats[N[Median[#[[4]]]/1000] & /@ Select[warm, #[[1]] == arm && #[[3]] === True && Length[#[[4]]] > 0 &]]], csArms];
  Do[cmp[other] = Association[Function[m, m -> <|
        "call" -> If[per["CUDAPP"]["arm_valid"] && per[other]["arm_valid"],
          csCall[per["CUDAPP"][m], per[other][m]], "NO_VERDICT"],
        "median_ratio_other_over_cudapp" -> N[per[other][m]["median"]/per["CUDAPP"][m]["median"]]|>] /@ Keys[csMetrics]],
    {other, {"PTXC", "NVRTC"}}];
  out["unit"] = "microseconds";
  out["arms"] = per;
  out["warm_build_per_process_median"] = warmPer;
  out["cudapp_vs"] = cmp;
  ExportString[out, "RawJSON", "Compact" -> True]];
