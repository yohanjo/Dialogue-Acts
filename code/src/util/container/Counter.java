package util.container;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.Vector;

import util.text.CSV;

public class Counter<K> extends TreeMap<K,Double> {
	
	public Counter () {
	}
	
	public Counter(Collection<? extends K> keys) {
		increaseAll(keys);
	}
	
	public void increase(K key) {
		Double value = this.get(key);
		if (value == null) this.put(key, 1.0);
		else this.put(key, value+1);
	}
	
	public void increase(K key, double n) {
		Double value = this.get(key);
		if (value == null) this.put(key, n);
		else this.put(key, value+n);
	}
	
	public void increaseAll(Collection<? extends K> keys) {
		for (K key : keys)
			increase(key);
	}
	
	public double sumAll() {
		double sum = 0;
		for (double value : values()) {
			sum += value;
		}
		return sum;
	}
	
	public void normalize() {
		double sum = sumAll();
		for (K key : keySet())
			put(key, get(key)/sum);
		return;
	}
	
	public void decrease(K key) throws Exception {
		Double value = this.get(key);
		if (value == null) throw new Exception("Error: no such element");
		else this.put(key, value-1);
	}
	
	public void decrease(K key, double n) throws Exception {
		Double value = this.get(key);
		if (value == null) throw new Exception("Error: no such element");
		else this.put(key, value-n);
	}
	
	public void removeIfLessThan(double threshold) {
		Vector<K> remove = new Vector<K>();
		for (Map.Entry<K,Double> entry : this.entrySet())
			if (entry.getValue() < threshold) remove.add(entry.getKey());
		
		for (K key : remove)
			this.remove(key);
	}
	
	public void removeIfGreaterThan(double threshold) {
		Vector<K> remove = new Vector<K>();
		for (Map.Entry<K,Double> entry : this.entrySet())
			if (entry.getValue() > threshold) remove.add(entry.getKey());
		
		for (K key : remove)
			this.remove(key);
	}

	public void retainSmallestValues(int num) {
		Vector<K> remove = new Vector<K>(); 
		TreeMap<Double, Vector<K>> invMap = getInverseMap();
		int cnt = 0;
		for (Double value : invMap.keySet()) {
			for (K key : invMap.get(value)) {
				if (++cnt > num) remove.add(key); 
			}
		}
		for (K key : remove) remove(key);
	}

	public void retainGreatestValues(int num) {
		Vector<K> remove = new Vector<K>(); 
		TreeMap<Double, Vector<K>> invMap = getInverseMap();
		int cnt = 0;
		for (Double value : invMap.navigableKeySet().descendingSet()) {
			for (K key : invMap.get(value)) {
				if (++cnt > num) remove.add(key); 
			}
		}
		for (K key : remove) remove(key);
	}
	
	public void add(Counter<K> c) {
		for (Map.Entry<K,Double> entry : c.entrySet()) {
			Double value = this.get(entry.getKey());
			if (value == null) this.put(entry.getKey(), entry.getValue());
			else this.put(entry.getKey(), value+entry.getValue());
		}
	}
	
	public double count(K key) {
		Double value = get(key);
		if (value == null) return 0;
		else return value;
	}
	
	public double average() {
		double sum = 0.0;
		for (double value : values()) sum += value;
		return sum / size();
	}
	
	public double stdDev() {
		double avg = average();
		double sum = 0.0;
		for (double value : values())
			sum += Math.pow( value-avg, 2 );
		return Math.sqrt( sum / size() );
	}
	
	public List<K> valueSortedKeys() {
		PriorityQueue<Map.Entry<K,Double>> sortedEntries = new PriorityQueue<Map.Entry<K,Double>>(this.size(), new Comparator<Map.Entry<K,Double>>() {
			@Override
			public int compare(Map.Entry<K,Double> o1, Map.Entry<K,Double> o2) {
				if (o1.getValue().compareTo(o2.getValue()) != 0) return o1.getValue().compareTo(o2.getValue());
				else return 1;
			}
		});
		for (Map.Entry<K,Double> entry : this.entrySet())
			sortedEntries.add(entry);
		
		Vector<K> sortedKeys = new Vector<K>(this.size());
		while (!sortedEntries.isEmpty())
			sortedKeys.add(sortedEntries.poll().getKey());
		
		return sortedKeys;
	}
	
	public TreeMap<Double, Vector<K>> getInverseMap() {
		TreeMap<Double, Vector<K>> sortedMap = new TreeMap<Double, Vector<K>>();
		for (Map.Entry<K, Double> entry : this.entrySet()) {
			Vector<K> keyList = sortedMap.get(entry.getValue());
			if (keyList == null) {
				keyList = new Vector<K>();
				sortedMap.put(entry.getValue(), keyList);
			}
			keyList.add(entry.getKey());
		}
		return sortedMap;
	}
	
	public String toCSVFormat() {
		String str = "";
		for (Map.Entry<K, Double> entry : this.entrySet()) {
			K key = entry.getKey();
			Double value = entry.getValue();
			if (key instanceof String) str += "\"" + key.toString().replaceAll("\"", "\"\"") + "\"," + value + "\n";
			else str += key + "," + value + "\n";
		}
		return str;
	}
	
	public void writeInCSVFormat(String outFilePath) throws Exception {
		PrintWriter outFile = new PrintWriter(new FileWriter(outFilePath));
		for (Map.Entry<K, Double> entry : this.entrySet()) {
			K key = entry.getKey();
			Double value = entry.getValue();
			outFile.println( CSV.quote(key.toString())+ "," + value);
		}
		outFile.close();
	}
	
	
	public String toCSVFormat(String delimitPattern) {
		String str = "";
		for (Map.Entry<K, Double> entry : this.entrySet()) {
			K key = entry.getKey();
			Double value = entry.getValue();
			if (key instanceof String) {
				String [] tokens = ((String) key).split(delimitPattern);
				for (String token : tokens) str += "\"" + token.replaceAll("\"", "\"\"") + "\",";
				str += value + "\n";
			}
			else str += key + "," + value + "\n";
		}
		return str;
	}
}
