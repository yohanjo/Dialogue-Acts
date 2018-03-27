package util.io;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

public class PrintFileWriter extends PrintWriter {

	public PrintFileWriter(String fileName) throws IOException {
		super(new FileWriter(fileName));
	}
	
	public PrintFileWriter(String fileName, String encoding) throws IOException {
		super(new OutputStreamWriter(new FileOutputStream(fileName), encoding));
	}
	
	public PrintFileWriter(String fileName, boolean append) throws IOException {
		super(new FileWriter(fileName, append));
	}
}
