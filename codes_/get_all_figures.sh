#!/bin/bash
uv run find_best_models.py "all_out_2"
for fig in $(ls figure_?.py); do
  uv run $fig "all_out_2"
done