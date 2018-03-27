package util.matrix;

import java.util.Arrays;

public class MatrixView {
    private DoubleMatrix mat = null;
    private double[][] diffs;
    private double[] rowSumDiffs;
    private boolean batchUpdate = true;
    
    public MatrixView(DoubleMatrix mat, boolean batchUpdate) {
        this.mat = mat;
        this.batchUpdate = batchUpdate;
        diffs = new double[mat.getNumRows()][mat.getNumColumns()];
        rowSumDiffs = new double[mat.getNumRows()];
    }
    
    public double getValue(int row, int col) {
        return mat.getValue(row, col) + diffs[row][col];
    }
    
    public double getRowSum(int row) {
        double diff = rowSumDiffs[row];
        return mat.getRowSum(row) + diff;
    }
    
    public void incValue(int row, int col) {
        incValue(row, col, 1);
    }
    
    public void incValue(int row, int col, double val) {
        if (batchUpdate) {
            diffs[row][col] += val;
            rowSumDiffs[row] += val;
        } else {
            mat.incValue(row, col, val);
        }
    }
    
    public void decValue(int row, int col) {
        decValue(row, col, 1);
    }
    
    public void decValue(int row, int col, double val) {
        incValue(row, col, -val);
    }
    
    public void commit() {
        for (int row = 0; row < mat.getNumRows(); row++) {
            for (int col = 0; col < mat.getNumColumns(); col++) {
                mat.incValue(row, col, diffs[row][col]);
            }
        }
        for (double[] rowDiffs : diffs)
            Arrays.fill(rowDiffs, 0);
        Arrays.fill(rowSumDiffs, 0);
    }
}
