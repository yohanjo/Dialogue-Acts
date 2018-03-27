package util.text;

public class CSV {

	public static String quote(String text) {
		return "\"" + text.replaceAll("\"", "\"\"") + "\"";
	}
	
	public static String unquote(String text) {
		text = text.replaceAll("^\"", "");
		text = text.replaceAll("\"$", "");
		text = text.replaceAll("\"\"", "\"");
		return text;
	}
}
