# Big Data Clustering

## Requirements

- Python 3.10
- Install all required packages using the provided requirements.txt file:

```
pip install -r requirements.txt
```

## Usage

1. **Generate cluster submissions**  
   ```
   python cluster_and_submit.py
   ```

2. **Create analysis visualizations**  
   ```
   python analysis_visualization.py
   ```

3. **Generate report**  
   This will create `Clustering_Report.docx` and `README.md` in the working directory:
   ```
   # The report is generated automatically by the above scripts
   ```

## Files

- `cluster_and_submit.py`: Clustering pipeline with Poisson-EM refinement.  
- `analysis_visualization.py`: Scripts to produce scatter plots, PCA/t-SNE, silhouette curves, and stability heatmap.  
- `Clustering_Report.docx`: This report document.  
- `README.md`: Usage instructions.
