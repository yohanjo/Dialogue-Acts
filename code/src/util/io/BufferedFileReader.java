package util.io;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class BufferedFileReader extends BufferedReader {

	String line = null;
	boolean lineBuffered = false;
	
	public BufferedFileReader(String fileName) throws Exception {
		super(new FileReader(fileName));
	}
	
	public BufferedFileReader(String fileName, String encoding) throws Exception {
		super(new InputStreamReader(new FileInputStream(fileName), encoding));
	}
	
	public boolean nextLine() throws IOException {
		if (lineBuffered) {
			if (line != null) return true;
			else return false;
		}
		else {
			line = super.readLine();
			lineBuffered = true;
			if (line != null) return true;
			else return false;
		}
	}
	
	@Override
	public String readLine() throws IOException {
		if (lineBuffered) {
			lineBuffered = false;
			return line;
		}
		else {
			line = super.readLine();
			lineBuffered = false;
			return line;
		}
	}
	
	public String readAll() throws IOException {
		String res = "";
		for (int i = 0; this.nextLine(); i++) {
			if (i > 0) res += "\n";
			res += this.readLine();
		}
		return res;
	}
}
