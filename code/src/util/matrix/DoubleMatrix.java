package util.matrix;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

import util.io.BufferedFileReader;


public class DoubleMatrix {
    private int numRows;
    private int numColumns;
    private double [][] rows;
    private double [] rowSum;
    private double [] colSum;
    
    ReadWriteLock lock = new ReentrantReadWriteLock();

    public DoubleMatrix(int numRows, int numColumns) {
        this.numRows = numRows;
        this.numColumns = numColumns;
        rows = new double[numRows][numColumns];
        rowSum = new double[numRows];
        colSum = new double[numColumns];
        reset();
    }

    @Deprecated
    public DoubleMatrix(int numOfRow, int numOfColumn, String fileType, String file) throws Exception{
        String line;
        int lineNumber=0;
        this.numRows = numOfRow;
        this.numColumns = numOfColumn;
        rows = new double[numOfRow][numOfColumn];
        rowSum = new double[numRows];
        colSum = new double[numColumns];

        BufferedReader fileReader = new BufferedReader(new FileReader(file));

        if(fileType.toLowerCase() == "normal") {
            while((line = fileReader.readLine()) != null) {
                String[] elements = line.split("\\s+");
                for(int j=0; j<elements.length;j++) {
                    rows[lineNumber][j] = Integer.valueOf(elements[j]);
                }
                lineNumber++;
            }
        }else if(fileType.toLowerCase() == "special") {
            for (int i = 0; i < numOfRow; i++)
                for (int j = 0; j < numOfColumn; j++)
                    rows[i][j] = 0;

            String docLine;
            while((line = fileReader.readLine()) != null) {
                docLine = fileReader.readLine();
                String[] elements = line.split("\\s+");
                int docWords = Integer.valueOf(elements[0]);

                String[] wordByCount = docLine.split("\\s+");

                if(docWords*2 != wordByCount.length) {
                    System.err.println("Matrix File Corrupted");
                    System.err.println(line);
                    System.exit(0);
                }

                for(int i = 0; i < docWords ; i++) {
                    rows[lineNumber][Integer.valueOf(wordByCount[2*i])] = Integer.valueOf(wordByCount[2*i+1]);
                }
                lineNumber++;
            }
        }
        fileReader.close();
    }

    public void reset() {
        for (int row = 0; row < numRows; row++)
            for (int col = 0; col < numColumns; col++)
                setValue(row, col, 0);
    }

    public DoubleMatrix transpose() {
        DoubleMatrix matrix = new DoubleMatrix(numColumns, numRows);
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numColumns; c++) {
                matrix.setValue(c, r, rows[r][c]);
            }
        }
        return matrix;
    }

    public void normalizeRows() {
        for (int r = 0; r < numRows; r++) {
            double sum = 0;
            for (int c = 0; c < numColumns; c++) 
                sum += getValue(r,c);
            for (int c = 0; c < numColumns; c++)
                setValue(r,c, getValue(r,c) / sum);
        }
    }

    public void normalizeCols() {
        for (int c = 0; c < numColumns; c++) {
            double sum = 0;
            for (int r = 0; r < numRows; r++)
                sum += getValue(r,c);
            for (int r = 0; r < numRows; r++)
                setValue(r,c, getValue(r,c) / sum);
        }
    }

    public int getNumRows() {
        return numRows;
    }

    public int getNumColumns() {
        return numColumns;
    }

    // set value of matrix(rowIdx, colIdx) to value
    public void setValue(int rowIdx, int colIdx, double value) {
        lock.writeLock().lock();
        double prevValue = getValue(rowIdx, colIdx);
        rowSum[rowIdx] += -prevValue + value;
        colSum[colIdx] += -prevValue + value;
        rows[rowIdx][colIdx] = value;	
        lock.writeLock().unlock();
    }

    // get value of matrix(rowIdx, colIdx)
    public double getValue(int rowIdx, int colIdx) {
        // lock.readLock().lock();
        double value = rows[rowIdx][colIdx]; 
        // lock.readLock().unlock();
        return value;
    }

    public void incValue(int rowIdx, int colIdx) {
        lock.writeLock().lock();
        rows[rowIdx][colIdx]++;
        rowSum[rowIdx]++;
        colSum[colIdx]++;
        lock.writeLock().unlock();
    }

    public void incValue(int rowIdx, int colIdx, double value) {
        lock.writeLock().lock();
        rows[rowIdx][colIdx] += value;
        rowSum[rowIdx] += value;
        colSum[colIdx] += value;
        lock.writeLock().unlock();
    }

    public void decValue(int rowIdx, int colIdx) {
        lock.writeLock().lock();
        rows[rowIdx][colIdx]--;
        rowSum[rowIdx]--;
        colSum[colIdx]--;
        lock.writeLock().unlock();
    }

    public double[] getRow(int rowIdx) {
        // lock.readLock().lock();
        double[] row = rows[rowIdx];
        // lock.readLock().unlock();
        return row;
    }

    // get sum of row elements
    public double getRowSum(int rowIdx) {
        // lock.readLock().lock();
        double value = rowSum[rowIdx];
        // lock.readLock().unlock();
        return value;
    }

    public double[] getColumn(int colIdx) {
        // lock.readLock().lock();
        double[] col = new double[numRows];
        for(int i=0; i<numRows; i++) {
            col[i] = rows[i][colIdx];
        }
        // lock.readLock().unlock();
        return col;
    }

    // get sum of column elements
    public double getColSum(int colIdx) throws Exception {
        // lock.readLock().lock();
        double value = colSum[colIdx];
        // lock.readLock().unlock();
        return value;
    }

    public double getSumAll() {
        // lock.readLock().lock();
        double sum = 0;
        for (int r = 0; r < numRows; r++)
            sum += rowSum[r];
        // lock.readLock().unlock();
        return sum;
    }

    public double [][] getMatrix() {
        return rows;
    }

    public IntegerMatrix getSortedIndexMatrix(int axis, int n) throws Exception {
        IntegerMatrix res;
        // lock.readLock().lock();
        if (axis==0) {  // Column-wise
            if (n > numRows) throw new Exception("n(="+n+") should be less than or equal to numRows(="+numRows+").");
            res = new IntegerMatrix(n, numColumns);
            for (int col = 0; col < numColumns; col++) {
                Vector<Integer> sortedIndex = getSortedColIndex(col, n);
                for (int row = 0; row < n; row++) res.setValue(row, col, sortedIndex.get(row));
            }
        }
        else {
            if (n > numColumns) throw new Exception("n(="+n+") should be less than or equal to numColumns(="+numColumns+").");
            res = new IntegerMatrix(numRows, n);
            for (int row = 0; row < numRows; row++) {
                Vector<Integer> sortedIndex = getSortedRowIndex(row, n);
                for (int col = 0; col < n; col++) res.setValue(row, col, sortedIndex.get(col));
            }
        }
        // lock.readLock().unlock();
        return res;
    }

    public Vector<Integer> getSortedRowIndex(int row, int n) {
        Vector<Integer> sortedList = new Vector<Integer>();
        // lock.readLock().lock();
        for(int i=0 ; i < n ; i++) {
            double maxValue = Integer.MIN_VALUE;
            int maxIndex = -1;
            for(int col=0 ; col<numColumns ; col++) {
                if(getValue(row, col) > maxValue) {
                    boolean exist = false;
                    for(int j=0 ; j<sortedList.size(); j++) {
                        if (sortedList.get(j) == col) {
                            exist = true;
                            break;
                        }
                    }
                    if(!exist) {
                        maxValue = getValue(row, col);
                        maxIndex = col;
                    }
                }
            }
            sortedList.add(maxIndex);
        }
        // lock.readLock().unlock();
        return sortedList;
    }

    public Vector<Integer> getSortedColIndex(int col, int n) {
        Vector<Integer> sortedList = new Vector<Integer>();
        // lock.readLock().lock();
        for(int i=0 ; i < n ; i++) {
            double maxValue = Integer.MIN_VALUE;
            int maxIndex = -1;
            for(int row=0 ; row<numRows ; row++) {
                if(getValue(row, col) > maxValue) {
                    boolean exist = false;
                    for(int j=0 ; j<sortedList.size(); j++) {
                        if (sortedList.get(j) == row) {
                            exist = true;
                            break;
                        }
                    }
                    if(!exist) {
                        maxValue = getValue(row, col);
                        maxIndex = row;
                    }
                }
            }
            sortedList.add(maxIndex);
        }
        // lock.readLock().unlock();
        return sortedList;
    }

    @Deprecated
    public void writeMatrixToCSVFile(String outputFilePath) throws Exception{
        PrintWriter out = new PrintWriter(new FileWriter(new File(outputFilePath)));

        for(int row=0; row < numRows ; row++) {
            for(int col=0; col < numColumns ; col++) {
                if(col == 0) out.print(getValue(row, col));
                else out.print(","+getValue(row, col));
            }
            out.println();
        }

        out.close();
    }

    public void saveToCsvFile(String outPath) throws Exception {
        PrintWriter out = new PrintWriter(new FileWriter(new File(outPath)));
        // lock.readLock().lock();
        for(int row=0; row < numRows ; row++) {
            out.println(
                Arrays.stream(getRow(row)).mapToObj(Double::toString)
                                          .collect(Collectors.joining(","))
            );
        }
        // lock.readLock().unlock();
        out.close();
    }
    
    public static DoubleMatrix loadFromCsvFile(String inPath) throws Exception {
        BufferedFileReader in = new BufferedFileReader(inPath);
        Integer numCols = null;
        Vector<String[]> strMatrix = new Vector<>();
        while (in.nextLine()) {
            String[] vals = in.readLine().split(",");
            if (numCols == null) numCols = vals.length;
            else assert(numCols == vals.length);
            strMatrix.add(vals);
        }
        in.close();
        
        int numRows = strMatrix.size();
        DoubleMatrix matrix = new DoubleMatrix(numRows, numCols);
        for (int row = 0; row < numRows; row++)
            for (int col = 0; col < numCols; col++)
                matrix.setValue(row, col, Double.valueOf(strMatrix.get(row)[col]));
        
        return matrix;
    }

    public DoubleMatrix copy() {
        DoubleMatrix temp = new DoubleMatrix(this.numRows, this.numColumns);
        // lock.readLock().lock();
        for(int row=0; row < numRows ; row++) {
            for(int col=0; col < numColumns ; col++) {
                temp.setValue(row, col, this.getValue(row, col));
            }
        }
        // lock.readLock().unlock();
        return temp;
    }

    public List<Double> getRowList(int row) {
        List<Double> list = new ArrayList<Double>();
        // lock.readLock().lock();
        for(int i=0; i<numColumns; i++) {
            list.add(getValue(row, i));
        }
        // lock.readLock().unlock();
        return list;
    }

    public List<Double> getColList(int col) {
        List<Double> list = new ArrayList<Double>();
        // lock.readLock().lock();
        for(int i=0; i<numRows; i++) {
            list.add(getValue(i, col));
        }
        // lock.readLock().unlock();
        return list;
    }


}
