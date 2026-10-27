(* PROTOCOL.md sections 5 and 7, A3(b). Evaluated in the Wolfram MCP kernel, one call per arm, with
   that arm's h2hRows (bench/extract.py) prepended. Row: {arm, round, metric, rep, batch, stream,
   load1, tSend, tFirst, tLast, promptTokens, completionTokens, busyRetries, ok}. Output: RawJSON. *)
(* A6(d)4: mean and standard deviation of the same values are added beside median/IQR. *)
h2hStats[v_List] := If[Length[v] == 0, <|"n" -> 0|>,
  <|"n" -> Length[v], "median" -> N[Median[v]], "q1" -> N[Quantile[v, 1/4]], "q3" -> N[Quantile[v, 3/4]],
    "iqr" -> N[InterquartileRange[v]], "p50" -> N[Quantile[v, 1/2]], "p99" -> N[Quantile[v, 99/100]],
    "mean" -> N[Mean[v]], "sd" -> If[Length[v] > 1, N[StandardDeviation[v]], Null]|>];
h2hAnalyse[rows_] := Module[{ok, bad, byMetric, perStream, perBatch, out = <||>},
  ok = Select[rows, #[[14]] === True &];
  bad = Counts[Select[rows, #[[14]] =!= True &][[All, 3]]];
  byMetric = GroupBy[ok, #[[3]] &];
  KeyValueMap[Function[{m, rs},
    Module[{batches = Values[GroupBy[rs, {#[[2]], #[[5]]} &]], ttft, dec, pre, tpot, agg},
      ttft = (#[[9]] - #[[8]]) & /@ rs;
      dec = Select[rs, #[[12]] > 1 &];
      tpot = ((#[[10]] - #[[9]])/(#[[12]] - 1)) & /@ dec;
      agg = (Total[#[[All, 12]]]/(Max[#[[All, 10]]] - Min[#[[All, 8]]])) & /@ batches;
      pre = (#[[11]]/(#[[9]] - #[[8]])) & /@ rs;
      out[m] = <|"rows_ok" -> Length[rs], "rows_failed" -> Lookup[bad, m, 0],
        "busy_retries_total" -> Total[rs[[All, 13]]],
        "ttft_s" -> h2hStats[ttft],
        "stream_decode_tok_s" -> h2hStats[1/tpot],
        "tpot_s" -> h2hStats[tpot],
        "batch_aggregate_tok_s" -> h2hStats[agg],
        "prefill_tok_s" -> h2hStats[pre]|>]], byMetric];
  KeyValueMap[(If[! KeyExistsQ[out, #1], out[#1] = <|"rows_ok" -> 0, "rows_failed" -> #2|>]) &, bad];
  ExportString[out, "RawJSON", "Compact" -> True]];
