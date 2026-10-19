# Story model comparison

Topics: 8 | same retrieved context per topic | temperature 0.65, num_ctx 6144

```
              runs  json_first_try  fallback_rate  fields_complete  check_valid  latency_s  story_words  fk_grade  grounding  terms_in_story
model                                                                                                                                       
gemma3:4b        8             1.0          0.000            1.000        1.000      24.75        276.0    10.538      0.896           0.688
qwen2.5:3b       8             1.0          0.125            0.982        0.250      24.50        153.0    10.329      0.833           0.635
llama3.1         8             1.0          0.000            1.000        1.000      91.55        268.5    10.475      0.958           0.896
qwen2.5:0.5b     8             1.0          0.625            0.804        0.375       5.45          0.0     8.300      0.143           0.000
gemma4:cloud     8             1.0          0.000            1.000        1.000       3.50        271.5    10.588      1.000           0.938
```
