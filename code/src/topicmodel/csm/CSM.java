/**
 * @author Yohan Jo
 * @version Aug 15, 2017
 */

package topicmodel.csm;

import java.io.File;
import java.io.FileReader;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.DoubleFunction;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;
import util.container.Counter;
import util.io.BufferedFileReader;
import util.io.PrintFileWriter;
import util.matrix.DoubleMatrix;
import util.matrix.IntegerMatrix;
import util.matrix.MatrixView;

public class CSM implements Callable<Void> {

    static int numStates;  // Num of states
    static int numFTopics;  // Num of foreground topics
    static int numBTopics;  // Num of background topics
    static int numWords;  //  Num of unique words

    static int numThreads = 1;
    static boolean useBatchParamUpdate;  // Update parameters in batch?
    static boolean updateParams;  // Update parameters?
    static boolean nuIsNotGiven;  // nu is not given?
    static boolean etaIsNotGiven;  // eta is not given?
    static boolean labelIsGiven;  // Label information is given?
    static boolean useDomain;  // Use the true domain information?
    static boolean useMultiLevel;  // Use the multi-level structure?
    static boolean tokenization;
    
    static int numIters;  // Num of training iterations
    static int numTmpIters;  // Num of iterations for temporary output files
    static int numLogIters;  // Num of iterations for computing log-likelihood
    static int burnIters;   // Num of iterations for burn-in for inference
    static int sampleInterval;  // Sampling interval
    static int modelIters = -1;  // Num of iterations for the input model
    
    static int numProbWords;  // Num of top words to display in the ProbWords file
    static String inDir=null;  // Input data directory
    static String outDir=null;  // Output directory
    static String dataFileName = null;  // Data file name (e.g., data.csv)
    static String modelPathPrefix = null;  // Trained model
    static String outPrefix = null;  // Prefix of the names of output files

    static DoubleMatrix N_SS;  // State x state transition counts
    static DoubleMatrix N_0S;  // Initial state counts with size (1, numStates)
    static DoubleMatrix N_FW;  // Foreground topic x word counts
    static DoubleMatrix N_BW;  // Background topic x word counts
    static DoubleMatrix N_SF;  // State x foreground topic counts
    static DoubleMatrix N_B;  // Background topic counts with size (1, numBTopics)
    static TreeMap<String, DoubleMatrix> N_QAS = new TreeMap<>();  // Seq x speaker x state counts
    
    static double fAlpha, sumFAlpha;
    static double bAlpha, sumBAlpha;
    static double beta, sumBeta;
    static double sGamma, sumSGamma;
    static double aGamma, sumAGamma;
    static double eta;
    static double nu;
    
    static Counter<String> wordCnt = new Counter<>();  // Vocab counts
    static TreeMap<String,Integer> wordIndex = new TreeMap<>();  // Vocab index
    static Vector<String> wordList = new Vector<>();  // Vocab list
    
    static TreeMap<String,Integer> domainIndex = new TreeMap<>();  // Domain index
    static Vector<String> domainList = new Vector<>();  // Domain list

    static TreeMap<String, Sequence> seqs = new TreeMap<>();  // Indexed seqs
    static TreeMap<String, Vector<String>> seqAuthorList = new TreeMap<>();  // Authors per seq
    static TreeMap<Integer,Double> logL = new TreeMap<>();  // Log-likelihood of the data
    static Vector<Duration> iterDurations = new Vector<>();

    static TreeMap<String, TreeMap<Integer,RawInstance>> rawSeqs = new TreeMap<>();
                                                                      // Non-indexed seqs
    static TreeMap<String, String> seqDomain = new TreeMap<>();  // Seq->domain mapping
    static TreeMap<String, TreeMap<Integer,Integer>> seqInstParent = 
                                    new TreeMap<>();  // Seq->Instance->parent mapping
    
    static TreeSet<String> stopwords = new TreeSet<String>();
    static int minNumWords;  // Minimum number of occurrences for a word to be included
    static int minSeqLen;  // Minimum length of a seq to be included

    
    // Thread-specific variables
    int threadId;
    List<String> seqIds;
    
    // Matrix views used by each thread. Keep track of within-thread changes.
    MatrixView V_SS;
    MatrixView V_0S;
    MatrixView V_FW;
    MatrixView V_BW;
    MatrixView V_SF;
    MatrixView V_B;


    // Command arguments
    public static class Parameters {
        @Parameter(names = "-s", description = "Number of states.")
        public int numStates = -1;

        @Parameter(names = "-ft", description = "Number of foreground topics.")
        public int numFTopics = -1;

        @Parameter(names = "-bt", description = "Number of background topics.")
        public int numBTopics = -1;

        @Parameter(names = "-i", description = "Number of iterations.", required=true)
        public int numIters = -1;

        @Parameter(names = "-to", description = "Number of iterations for temporary output files.")
        public int numTmpIters = -1;

        @Parameter(names = "-log", description = "Number of iterations for calculating "
                                                                    + " log-likelihood.")
        public int numLogIters = -1;
        
        @Parameter(names = "-burn", description = "Number of iterations for burn-in.")
        public int burnIters = 0;
        
        @Parameter(names = "-sample", description = "Sampling interval.")
        public int sampleInterval = -1;

        @Parameter(names = "-pw", description = "Number of top words to display.")
        public int numProbWords = 100;

        @Parameter(names = "-fa", description = "alpha^F.")
        public double fAlpha = -1;

        @Parameter(names = "-ba", description = "alpha^B.")
        public double bAlpha = -1;

        @Parameter(names = "-b", description = "beta.")
        public double beta = -1;

        @Parameter(names = "-sg", description = "gamma^S.")
        public double sGamma = -1;

        @Parameter(names = "-ag", description = "gamma^A.")
        public double aGamma = -1;

        @Parameter(names = "-n", description = "nu.")
        public double nu = -1;

        @Parameter(names = "-e", description = "eta.")
        public double eta = -1;

        @Parameter(names = "-d", description = "Input directory.", required=true)
        public String inDir = null;

        @Parameter(names = "-o", description = "Output directory.")
        public String outDir = null;

        @Parameter(names = "-data", description = "Data file name.", required=true)
        public String dataFileName = null;

        @Parameter(names = "-sw", description = "Stopwords file name.")
        public String stopwordsFileName = null;

        @Parameter(names = "-mw", description = "Minimum number of words to include.")
        public int minNumWords = 1;

        @Parameter(names = "-ms", description = "Minimum length of sequences to include.")
        public int minSeqLen = 1;

        @Parameter(names = "-tok", description = "Do tokenization/pattern replacement.")
        public boolean tokenize;

        @Parameter(names = "-seq", description = "Assume a sequential dialogue and do not "
                                    + "use the parent information. If this flag is not set, "
                                    + "the data file is expected to have the Parent column.")
        public boolean seq;
        
        @Parameter(names = "-model", description = "Trained model to fit (no parameter update).")
        public String model = null;

        @Parameter(names = "-domain", description = "Use the domain information.")
        public boolean useDomain;

        @Parameter(names = "-th", description = "Number of threads.")
        public int numThreads = 1;

        @Parameter(names = "-help", description = "Command description.", help=true)
        public boolean help;
    }

    public static void main (String [] args) throws Exception {
        // Load command arguments
        Parameters params = new Parameters();
        JCommander jc = JCommander.newBuilder().addObject(params).build();
        jc.parse(args);
        if (params.help) {
            jc.usage();
            System.exit(0);
        }

        if (params.model != null) {
            updateParams = false;
            modelPathPrefix = params.model;
        } else {
            updateParams = true;
        }
        
        numThreads = params.numThreads;
        if (numThreads > 1) useBatchParamUpdate = true;
        else useBatchParamUpdate = false;
        numStates = params.numStates;
        numFTopics = params.numFTopics;
        numBTopics = params.numBTopics;
        numIters = params.numIters;
        numTmpIters = params.numTmpIters;
        numLogIters = params.numLogIters;
        burnIters = params.burnIters;
        sampleInterval = params.sampleInterval;
        numProbWords = params.numProbWords;
        
        inDir = params.inDir;
        outDir = params.outDir;
        dataFileName = params.dataFileName;

        minNumWords = params.minNumWords;
        tokenization = params.tokenize;
        
        useDomain = params.useDomain;
        useMultiLevel = !params.seq;
        minSeqLen = params.minSeqLen;
        
        fAlpha = params.fAlpha;
        bAlpha = params.bAlpha;
        beta = params.beta;
        sGamma = params.sGamma;
        aGamma = params.aGamma;
        eta = params.eta;
        nu = params.nu;
        
        if (modelPathPrefix != null) {
            for (String argToken : modelPathPrefix.split("-|/")) {
                Matcher m = Pattern.compile("^(.*?)([\\d.]*)$").matcher(argToken);
                if (!m.find()) continue;
                String arg = m.group(1);
                String val = m.group(2);
                
                if (arg.equals("S")) numStates = Integer.valueOf(val);
                else if (arg.equals("FT")) numFTopics = Integer.valueOf(val);
                else if (arg.equals("BT")) numBTopics = Integer.valueOf(val);
                else if (arg.equals("FA")) fAlpha = Double.valueOf(val);
                else if (arg.equals("BA")) bAlpha = Double.valueOf(val);
                else if (arg.equals("B")) beta = Double.valueOf(val);
                else if (arg.equals("SG")) sGamma = Double.valueOf(val);
                else if (arg.equals("AG")) aGamma = Double.valueOf(val);
                else if (arg.equals("E")) eta = Double.valueOf(val);
                else if (arg.equals("N")) nu = Double.valueOf(val);
                else if (arg.equals("MS")) minSeqLen = Integer.valueOf(val);
                else if (arg.equals("DOM")) useDomain = true;
                else if (arg.equals("SEQ")) useMultiLevel = false;
                else if (arg.equals("I")) modelIters = Integer.valueOf(val);
            }
        }
        
        
        nuIsNotGiven = (nu == -1 ? true : false);
        etaIsNotGiven = (eta == -1 ? true : false);
        
        
        // Validity
        if (!new File(inDir).exists()) throw new Exception("There's no input directory " + inDir);
        if (fAlpha <= 0) throw new Exception("alpha^F must be specified as a positive real number.");
        if (bAlpha <= 0) throw new Exception("alpha^B must be specified as a positive real number.");
        if (beta <= 0) throw new Exception("beta must be specified as a positive real number.");
        if (sGamma <= 0) throw new Exception("gamma^S must be specified as a positive real number.");
        if (aGamma <= 0) throw new Exception("gamma^A must be specified as a positive real number.");
        if (numBTopics <= 0 && !useDomain) throw new Exception("Either the number of BTs should "
                                    + " be specified or the domain information should be used.");
        if (numBTopics > 0 && useDomain) System.err.println("Warning: the argument -bt is ignored "
                                    + " because the domain information is used.");
        
        
        // Training
        if (updateParams) {
            loadStopwords(params.stopwordsFileName);
            loadInstances();
            indexizeWords();
            if (useDomain) indexizeDomains();
            indexizeInstances();
            buildSeqs();
            
            setUpOutputEnvironment();
            printConfiguration();

            saveWords();
            if (useDomain) saveDomains();
            
            runSampling();
            
        // Fitting (no parameter update)
        } else {
            loadWords();
            if (useDomain) loadDomains();
            loadInstances();
            indexizeInstances();
            buildSeqs();
            
            setUpOutputEnvironment();
            printConfiguration();

            saveWords();
            
            runSampling();
        }
    }
    
    /**
     * Prints the model configuration.
     */
    public static void printConfiguration() {
        System.out.println("Threads: " + numThreads);
        System.out.println("Use batch param update: " + useBatchParamUpdate);
        System.out.println("Use multi-level: " + useMultiLevel);
        System.out.println("Use domains: " + useDomain);
        System.out.println("Nu is given: " + !nuIsNotGiven);
        System.out.println("Eta is given: " + !etaIsNotGiven);
        System.out.println("Domains: " + numBTopics);
        System.out.println("Foreground topics: " + numFTopics);
        System.out.println("Words: " + numWords);
        System.out.println("Seqs: " + seqs.size());
        System.out.println("Min seq len: " + minSeqLen);
    }
    
    /**
     * Constructs an instance of this class for running a thread.
     */
    public CSM(int threadId, List<String> seqIds) {
        this.threadId = threadId;
        this.seqIds = seqIds;
        
        // Make matrix views to keep track of within-thread changes.
        V_SS = new MatrixView(N_SS, useBatchParamUpdate);
        V_0S = new MatrixView(N_0S, useBatchParamUpdate);
        V_FW = new MatrixView(N_FW, useBatchParamUpdate);
        V_BW = new MatrixView(N_BW, useBatchParamUpdate);
        V_SF = new MatrixView(N_SF, useBatchParamUpdate);
        V_B = new MatrixView(N_B, useBatchParamUpdate);
    }
    
    /**
     * Samples the parameters for one iteration and 
     * updates the within-thread counters to the global counters.
     */
    @Override
    public Void call() throws Exception {
        int beginThread = threadId * seqIds.size() / numThreads;
        int endThread = (threadId+1) * seqIds.size() / numThreads;
        List<String> subSeqIds = seqIds.subList(beginThread, endThread); 

        try {
            sampleFTopic(subSeqIds);
            sampleLevel(subSeqIds);
            if (!useDomain) sampleBTopic(subSeqIds);
            sampleState(subSeqIds);

            // Commit within-thread changes and update the global matrix.
            V_SS.commit();
            V_0S.commit();
            V_FW.commit();
            V_BW.commit();
            V_SF.commit();
            V_B.commit();
            
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }

        return null;
    }
   
    /**
     * Runs Gibbs sampling. For each iteration, this method shuffles the order 
     * of the sequences, splits and assigns the sequences into threads,
     * and invokes and joins the threads.
     */
    public static void runSampling() throws Exception {
        initialize();
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        Vector<String> seqIds = new Vector<String>(seqs.keySet());
        
        for (int iter = 0; iter < numIters; iter++) {
            System.out.print("Iteration "+iter);
            LocalDateTime startTime = LocalDateTime.now();

            // Randomize the order of sequences.
            // Make sure previous threads have been finished before calling this.
            Collections.shuffle(seqIds);
            
            List<Callable<Void>> tasks = new Vector<Callable<Void>>();
            for (int tid = 0; tid < numThreads; tid++) {
                tasks.add(new CSM(tid, seqIds));
            }
            executor.invokeAll(tasks);
            
            // Update nu and eta
            if (updateParams && nuIsNotGiven) sampleNu();
            if (updateParams && etaIsNotGiven) sampleEta();

            Duration runTime = Duration.between(startTime, LocalDateTime.now());
            Duration remainTime = runTime.multipliedBy(numIters - iter - 1);
            System.out.println(String.format(" took %.3fs. (Remain: %dH %2dM %2dS)",
                                runTime.toMillis()/1000.0, remainTime.toHours(), 
                                remainTime.toMinutes()%60, remainTime.getSeconds()%60));

            if (numLogIters > 0 && (iter+1) % numLogIters == 0) {
                double ll = logLikelihood();
                System.out.println("  - logP: "+ll);
                logL.put(iter+1, ll);
            }
            if (iter+1 == numIters || (numTmpIters > 0 && (iter+1) % numTmpIters == 0)) {
                System.out.println("  - Generating output files...");
                genOutFiles(iter+1);
            } else if (sampleInterval > 0 && iter+1 > burnIters && 
                    (iter+1-burnIters) % sampleInterval == 0) {
                printInstAssign(iter+1);
            }
        }
        executor.shutdownNow();
    }
    
    /**
     * Generates the output file name prefix and necessary output directories.
     */
    public static void setUpOutputEnvironment() throws Exception {
        outPrefix = "CSM-" 
                    + dataFileName.replace(".csv","")
                    + (minSeqLen > 1  ? "-MS"+minSeqLen : "")
                    + "-S" + numStates
                    + "-FT" + numFTopics
                    + "-BT" + numBTopics
                    + "-FA" + fAlpha
                    + "-BA" + bAlpha
                    + "-B" + beta
                    + "-SG" + sGamma
                    + "-AG" + aGamma
                    + (!etaIsNotGiven ? "-E" + eta : "")
                    + (!nuIsNotGiven ? "-N" + nu : "")
                    + (useDomain ? "-DOM" : "")
                    + (!useMultiLevel ? "-SEQ" : "")
                    + (modelIters != -1 ? "-I" + modelIters : "");
        if (outDir == null) outDir = inDir+"/"+outPrefix;
        Files.createDirectories(Paths.get(outDir));
    }

    /**
     * Loads stop words from the given file.
     * 
     * @param stopwordsFileName the name of the stop words file.
     */
    public static void loadStopwords(String stopwordsFileName) throws Exception {
        if (stopwordsFileName != null) {
            BufferedFileReader stopwordsFile = new BufferedFileReader(inDir+"/"+stopwordsFileName);
            while (stopwordsFile.nextLine()) {
                stopwords.add(stopwordsFile.readLine().trim());
            }
            stopwordsFile.close();
        }
    }

    /**
     * Loads instances from the data file. The loaded data is stored in
     * temporary data structure {@link #rawSeqs}, which is later indexized 
     * in {@link #indexizeInstances()}.
     */
    public static void loadInstances() throws Exception {
        System.out.println("Loading data...");

        CSVParser inData = new CSVParser(
                new FileReader(inDir+"/"+dataFileName), CSVFormat.EXCEL.withHeader());
        
        labelIsGiven = inData.getHeaderMap().containsKey("Label");

        for (CSVRecord record : inData) {
            String seqId = record.get("SeqId");
            int instNo = Integer.valueOf(record.get("InstNo"));
            int parent = (useMultiLevel ? Integer.valueOf(record.get("Parent")) : -1);
            String author = record.get("Author");

            if (useDomain) {
                seqDomain.putIfAbsent(seqId, record.get("Domain"));
            }
            if (useMultiLevel) {
                seqInstParent.putIfAbsent(seqId, new TreeMap<Integer,Integer>());
                seqInstParent.get(seqId).put(instNo, parent);
            }

            RawInstance inst = new RawInstance();

            if (labelIsGiven) inst.label = record.get("Label");
            inst.text = record.get("Text");
            if (tokenization) {
                inst.text = inst.text.replaceAll("https?://\\S+", "URLURL");
                inst.text = inst.text.replaceAll("[\\w.]+@[\\w.]+", "EMAILEMAIL");
                DocumentPreprocessor dp = new DocumentPreprocessor(new StringReader(inst.text));
                for (List<HasWord> sentence : dp) {
                    Vector<String> words = new Vector<String>();
                    for (HasWord word : sentence) {
                        if (word.word().length()==0) continue;
                        String w = word.word().toLowerCase();
                        if (w.matches("^[\\d]+$")) w = "##NUMBER##";
                        if (stopwords.contains(w)) continue;

                        words.add(w);
                        if (updateParams) wordCnt.increase(w);
                    }
                    if (!words.isEmpty()) inst.sentences.add(words);
                }
            } else {
                Vector<String> words = new Vector<String>();
                for (String w : inst.text.split(" ")) {
                    if (w.equals("<SENT>") && !words.isEmpty()) {
                        inst.sentences.add(words);
                        words = new Vector<>();
                    } else {
                        if (w.length()==0) continue;
                        if (stopwords.contains(w)) continue;
    
                        words.add(w);
                        if (updateParams) wordCnt.increase(w);
                    }
                }
                if (!words.isEmpty()) inst.sentences.add(words);
            }

            if (!inst.sentences.isEmpty()) {
                inst.author = author;
                rawSeqs.putIfAbsent(seqId, new TreeMap<Integer,RawInstance>());
                rawSeqs.get(seqId).put(instNo, inst);
            }
        }
        inData.close();
    }

    /**
     * Indexizes the words loaded from the data file and generates 
     * {@link #wordIndex} and {@link #wordList}.
     */
    public static void indexizeWords() {
        wordIndex = new TreeMap<String,Integer>();
        wordList = new Vector<String>();
        for (String word : wordCnt.keySet()) {
            if (minNumWords > 0 && wordCnt.get(word) < minNumWords) continue;
            wordIndex.put(word, wordIndex.size());
            wordList.add(word);
        }
        numWords = wordIndex.size();
    }

    /**
     * Indexizes the loaded instances by changing the words and authors with
     * the corresponding indices. This method generates {@link #seqs}.
     */
    public static void indexizeInstances() {
        for (String seqId : rawSeqs.keySet()) {
            TreeMap<Integer,RawInstance> rawSeq = rawSeqs.get(seqId);
            Sequence seq = new Sequence();
            TreeMap<String,Integer> authorIndex = new TreeMap<String,Integer>();
            for (int instNo : rawSeq.keySet()) {
                RawInstance rawInst = rawSeq.get(instNo);
                Instance inst = new Instance();
                for (Vector<String> rawSentence : rawInst.sentences) {
                    Sentence sentence = new Sentence();
                    for (String word : rawSentence) {
                        Integer index = wordIndex.get(word);
                        if (index != null) sentence.words.add(new Word(index));
                    }
                    if (!sentence.words.isEmpty()) inst.sentences.add(sentence);
                }
                if (!inst.sentences.isEmpty()) {
                    authorIndex.putIfAbsent(rawInst.author, authorIndex.size());
                    inst.author = authorIndex.get(rawInst.author);
                    seq.instances.put(instNo, inst);
                }
            }
            if (seq.instances.size() >= minSeqLen) {
                seqs.put(seqId, seq);
                String [] authorArray = new String[authorIndex.size()];
                for (Map.Entry<String,Integer> entry : authorIndex.entrySet())
                    authorArray[entry.getValue()] = entry.getKey();  // index, author
                Vector<String> authorList = new Vector<String>(Arrays.asList(authorArray));
                seqAuthorList.put(seqId, authorList);
            }
        }
    }

    /**
     * Connects instances either with parent information or by time order. 
     */ 
    public static void buildSeqs() {
        for (String seqId : seqs.keySet()) {
            TreeMap<Integer,Instance> insts = seqs.get(seqId).instances;
            int prevInstNo = -1;
            for (Map.Entry<Integer,Instance> entry : insts.entrySet()) {
                int instNo = entry.getKey();
                Instance inst = entry.getValue();
                Instance parent = null;
                if (useMultiLevel) parent = insts.get(seqInstParent.get(seqId).get(instNo));
                else parent = insts.get(prevInstNo);
                if (parent != null) {
                    inst.parent = parent;
                    parent.children.add(inst);    
                }
                prevInstNo = instNo;
            }
        }
    }

    /**
     * Indexizes the domains (i.e., Wikipedia article titles) and 
     * generates {@link #domainIndex} and {@link #domainList}.
     */
    public static void indexizeDomains() {
        TreeSet<String> domainSet = new TreeSet<>(seqDomain.values());
        for (String domain : domainSet) {
            domainIndex.put(domain, domainIndex.size());
            domainList.add(domain);
        }
        numBTopics = domainIndex.size();
    }

    /**
     * Initializes all parameters and variables.
     */
    public static void initialize() throws Exception {
        Random random = new Random();

        if (updateParams) {
            N_SS = new DoubleMatrix(numStates, numStates);
            N_0S = new DoubleMatrix(1, numStates);
            N_FW = new DoubleMatrix(numFTopics, numWords);
            N_BW = new DoubleMatrix(numBTopics, numWords);
            N_SF = new DoubleMatrix(numStates, numFTopics);
            N_B = new DoubleMatrix(1, numBTopics);
            
            if (nuIsNotGiven) nu = random.nextDouble();
            if (etaIsNotGiven) eta = random.nextDouble();
        } else {
            restoreModel();
        }
        
        N_QAS = new TreeMap<String, DoubleMatrix>();
        for (String seqId : seqAuthorList.keySet())
            N_QAS.put(seqId, new DoubleMatrix(seqAuthorList.get(seqId).size(), numStates));

        sumFAlpha = numFTopics * fAlpha;
        sumBAlpha = numBTopics * bAlpha;
        sumBeta = numWords * beta;
        sumSGamma = numStates * sGamma;
        sumAGamma = numStates * aGamma;


        // Initialize instances
        for (String seqId : seqs.keySet()) {
            Sequence seq = seqs.get(seqId);
            if (useDomain) {
                seq.bTopic = domainIndex.getOrDefault(seqDomain.get(seqId), 
                                                        random.nextInt(numBTopics));
            } else {
                seq.bTopic = random.nextInt(numBTopics);
            }
            if (updateParams) N_B.incValue(0, seq.bTopic);

            Vector<Instance> insts = new Vector<Instance>(seq.instances.values());
            for (int d = 0; d < insts.size(); d++) {
                Instance inst = insts.get(d);
                inst.state = random.nextInt(numStates);
                N_QAS.get(seqId).incValue(inst.author, inst.state);

                for (Sentence sentence : inst.sentences) {
                    sentence.fTopic = random.nextInt(numFTopics);
                    if (updateParams) N_SF.incValue(inst.state, sentence.fTopic);

                    for (Word word : sentence.words) {
                        if (random.nextDouble() < eta) {
                            word.level = 1;
                            if (updateParams) N_FW.incValue(sentence.fTopic, word.id);
                        }
                        else {
                            word.level = 0;
                            if (updateParams) N_BW.incValue(seq.bTopic, word.id);
                        }
                    }
                }

                if (inst.parent == null) {
                    if (updateParams) N_0S.incValue(0, inst.state);
                }
                else {
                    if (updateParams) N_SS.incValue(inst.parent.state, inst.state);
                }
            }
        }
    }

    /**
     * Samples the foreground topics of the given sequences. 
     *
     * @param seqIds the sequences whose foreground topics are sampled.
     */
    public void sampleFTopic(List<String> seqIds) throws Exception {
        for (String seqId : seqIds) {
            Sequence seq = seqs.get(seqId);
            Vector<Instance> insts = new Vector<Instance>(seq.instances.values());
            for (int d = 0; d < insts.size(); d++) {
                Instance inst = insts.get(d);
                int state = inst.state;

                for (Sentence sentence : inst.sentences) {
                    int oldTopic = sentence.fTopic;

                    // Remove the old assignment
                    if (updateParams) {
                        V_SF.decValue(state, oldTopic);
                        for (Word word : sentence.words) {
                            if (word.level==1) V_FW.decValue(oldTopic, word.id);
                        }
                    }

                    // Calculate the conditional probability of topic assignment
                    double [] probs = new double[numFTopics];
                    for (int t = 0; t < numFTopics; t++) {
                        probs[t] += Math.log(V_SF.getValue(state, t) + fAlpha);
                        for (Word word : sentence.words) {
                            if (word.level==1) {
                                probs[t] += Math.log((V_FW.getValue(t,word.id) + beta) 
                                                        / (V_FW.getRowSum(t) + sumBeta));
                                if (updateParams) V_FW.incValue(t, word.id);
                            }
                        }
                        // Revert the counter
                        if (updateParams) {
                            for (Word word : sentence.words) {
                                if (word.level==1) V_FW.decValue(t, word.id);
                            }
                        }
                    }

                    double maxLogP = max(probs);
                    for (int t = 0; t < numFTopics; t++)
                        probs[t] = Math.exp(probs[t] - maxLogP);

                    int newTopic = sampleIndex(probs);
                    sentence.fTopic = newTopic;
                    
                    // Update the new assignment
                    if (updateParams) {
                        V_SF.incValue(state, newTopic);
                        for (Word word : sentence.words) {
                            if (word.level==1) V_FW.incValue(newTopic, word.id);
                        }
                    }
                }
            }
        }
    }

    /**
     * Samples the background topics of the given sequences. 
     *
     * @param seqIds the sequences whose background topics are sampled.
     */
    public void sampleBTopic(List<String> seqIds) throws Exception {
        for (String seqId : seqIds) {
            Sequence seq = seqs.get(seqId);
            int oldTopic = seq.bTopic;

            // Remove the old assignment
            if (updateParams) {
                V_B.decValue(0, oldTopic);
                for (Instance inst : seq.instances.values()) {
                    for (Sentence sentence : inst.sentences) {
                        // Remove old topic assignment for this sentence
                        for (Word word : sentence.words) {
                            if (word.level==1) continue;
                            V_BW.decValue(oldTopic, word.id);
                        }
                    }
                }
            }

            // Calculate the conditional probability of topic assignment
            double [] probs = new double[numBTopics];
            for (int t = 0; t < numBTopics; t++) {
                probs[t] += Math.log(V_B.getValue(0,t) + bAlpha);
                for (Instance inst : seq.instances.values()) {
                    for (Sentence sentence : inst.sentences) {
                        for (Word word : sentence.words) {
                            if (word.level==1) continue;
                            int w = word.id;
                            probs[t] += Math.log(V_BW.getValue(t,w) + beta) 
                                        - Math.log(V_BW.getRowSum(t) + sumBeta);
                            if (updateParams) V_BW.incValue(t,w);
                        }
                    }
                }
                // Revert the counters
                if (updateParams) {
                    for (Instance inst : seq.instances.values()) {
                        for (Sentence sentence : inst.sentences) {
                            for (Word word : sentence.words) {
                                if (word.level==1) continue;
                                V_BW.decValue(t,word.id);
                            }
                        }
                    }
                }
            }	

            double maxLogP = max(probs);
            for (int t = 0; t < numBTopics; t++)
                probs[t] = Math.exp(probs[t] - maxLogP);

            int newTopic = sampleIndex(probs);
            seq.bTopic = newTopic;

            // Update the new assignment
            if (updateParams) {
                V_B.incValue(0, newTopic);
                for (Instance inst : seq.instances.values()) {
                    for (Sentence sentence : inst.sentences) {
                        for (Word word : sentence.words) {
                            if (word.level==1) continue;
                            V_BW.incValue(newTopic, word.id);
                        }
                    }
                }
            }
        }
    }

    /**
     * Samples the level ("foreground" or "background") of each word 
     * in the given sequences. 
     *
     * @param seqIds the sequences for which the level of each word is sampled.
     */
    public void sampleLevel(List<String> seqIds) throws Exception {
        for (String seqId : seqIds) {
            Sequence seq = seqs.get(seqId);
            int bTopic = seq.bTopic;
            for (Instance inst : seq.instances.values()) {
                for (Sentence sentence : inst.sentences) {
                    int fTopic = sentence.fTopic;
                    for (Word word : sentence.words) {
                        int oldLevel = word.level;
                        int w = word.id;

                        // Remove the old assignment
                        if (updateParams) {
                            if (oldLevel==0) V_BW.decValue(bTopic, w);
                            else V_FW.decValue(fTopic, w);
                        }

                        double [] probs = new double[2];
                        probs[0] = (1-eta) * (V_BW.getValue(bTopic,w) + beta) 
                                            / (V_BW.getRowSum(bTopic) + sumBeta);
                        probs[1] = eta * (V_FW.getValue(fTopic, w) + beta)
                                            / (V_FW.getRowSum(fTopic) + sumBeta);

                        int newLevel = sampleIndex(probs);
                        word.level = newLevel;
                        
                        // Update the new assignment
                        if (updateParams) {
                            if (newLevel==0) V_BW.incValue(bTopic, w);
                            else V_FW.incValue(fTopic, w);
                        }
                    }
                }
            }
        }
    }

    /**
     * Samples the states of the given sequences. 
     *
     * @param seqIds the sequences whose states are sampled.
     */
    public void sampleState(List<String> seqIds) throws Exception {
        for (String seqId : seqIds) {
            DoubleMatrix N_AS = N_QAS.get(seqId);
            Sequence seqVal = seqs.get(seqId);
            Vector<Instance> seq = new Vector<Instance>(seqVal.instances.values());
            for (int d = 0; d < seq.size(); d++) {
                Instance inst = seq.get(d);
                int oldState = inst.state;

                // Remove the old assignment
                N_AS.decValue(inst.author, oldState);
                if (updateParams) {
                    for (Sentence sentence : inst.sentences)
                        V_SF.decValue(oldState, sentence.fTopic);
                    if (inst.parent == null) V_0S.decValue(0, oldState);
                    else V_SS.decValue(inst.parent.state, oldState);
                    for (Instance nextInst : inst.children) {
                        V_SS.decValue(oldState, nextInst.state);
                    }
                }

                // Calculate the conditional probability of state assignment
                double [] probs = new double[numStates];
                for (int s = 0; s < numStates; s++) {
                    // p(FT | s)
                    for (Sentence sentence : inst.sentences) {
                        int fTopic = sentence.fTopic;
                        probs[s] += Math.log((V_SF.getValue(s, fTopic) + fAlpha) 
                                                / (V_SF.getRowSum(s) + sumFAlpha));
                        if (updateParams) V_SF.incValue(s, fTopic);
                    }
                    // Revert the counter
                    if (updateParams) {
                        for (Sentence sentence : inst.sentences) {
                            V_SF.decValue(s, sentence.fTopic);
                        }
                    }

                    // p(s | s_prev, a)
                    if (inst.parent == null) {  // Root post
                        probs[s] += Math.log(nu * (V_0S.getValue(0,s) + sGamma) 
                                                / (V_0S.getRowSum(0) + sumSGamma)
                                    + (1-nu) * (N_AS.getValue(inst.author,s) + aGamma) 
                                                / (N_AS.getRowSum(inst.author) + sumAGamma));
                    } else {
                        probs[s] += Math.log(nu * (V_SS.getValue(inst.parent.state, s) + sGamma) 
                                                / (V_SS.getRowSum(inst.parent.state) + sumSGamma)
                                    + (1-nu) * (N_AS.getValue(inst.author,s) + aGamma) 
                                                / (N_AS.getRowSum(inst.author) + sumAGamma));
                        if (updateParams) V_SS.incValue(inst.parent.state, s);
                    }
                    N_AS.incValue(inst.author, s);

                    // p(s_next | s, a_next)
                    for (Instance nextInst : inst.children) {
                        probs[s] += Math.log(nu * (V_SS.getValue(s, nextInst.state) + sGamma) 
                                                / (V_SS.getRowSum(s) + sumSGamma)
                                    + (1-nu) * (N_AS.getValue(nextInst.author, nextInst.state) + aGamma)
                                                / (N_AS.getRowSum(nextInst.author) + sumAGamma));
                        if (updateParams) V_SS.incValue(s, nextInst.state);
                    }

                    // Revert the counters
                    N_AS.decValue(inst.author, s);
                    if (updateParams) {
                        if (inst.parent != null) {  // Previous state exists
                            V_SS.decValue(inst.parent.state, s);
                        }
                        for (Instance nextInst : inst.children) {
                            V_SS.decValue(s, nextInst.state);
                        }
                    }
                }

                double maxLogP = max(probs);
                for (int s = 0; s < numStates; s++)
                    probs[s] = Math.exp(probs[s] - maxLogP);

                int newState = sampleIndex(probs);
                inst.state = newState;

                // Update the new assignment
                N_AS.incValue(inst.author, newState);
                if (updateParams) {
                    for (Sentence sentence : inst.sentences)
                        V_SF.incValue(newState, sentence.fTopic);
                    if (inst.parent == null) V_0S.incValue(0, newState);
                    else V_SS.incValue(inst.parent.state, newState);
                    for (Instance nextInst : inst.children)
                        V_SS.incValue(newState, nextInst.state);
                }
            }
        }
    }
    
    /**
     * Samples nu.
     */
    public static void sampleNu() {
        // PDF = P(all transitions between states)
        nu = sliceSampling((x) -> {
            double logP = 0;
            for (String seqKey : seqs.keySet()) {
                DoubleMatrix N_AS = N_QAS.get(seqKey);
                Sequence seq = seqs.get(seqKey);
                for (Instance inst : seq.instances.values()) {
                    int author = inst.author;
                    int state = inst.state;
                    if (inst.parent == null) {
                        logP += Math.log(x * (N_0S.getValue(0, state) + sGamma) / (N_0S.getRowSum(0) + sumSGamma)
                                + (1 - x) * (N_AS.getValue(author, state) + aGamma) / (N_AS.getRowSum(author) + sumAGamma));
                    } else {
                        logP += Math.log(x * (N_SS.getValue(inst.parent.state, state) + sGamma) / (N_SS.getRowSum(inst.parent.state) + sumSGamma)
                                + (1 - x) * (N_AS.getValue(author, state) + aGamma) / (N_AS.getRowSum(author) + sumAGamma));    
                    }
                }
            }
            return logP;
        }, nu);
    }
    
    /**
     * Samples eta.
     */
    public static void sampleEta() {
        // Count observations
        int[] levelCnt = new int[2];
        for (Sequence seq : seqs.values()) {
            for (Instance inst : seq.instances.values()) {
                for (Sentence sentence : inst.sentences) {
                    for (Word word : sentence.words) {
                        levelCnt[word.level]++;
                    }
                }
            }
        }
        
        // PDF = Binomial(observation 1, observation 2, p(observation 1))
        eta = sliceSampling((x) -> levelCnt[1] * Math.log(x) + levelCnt[0] * Math.log(1 - x), eta);
    }

    
    /**
     * Runs slice sampling and returns one sample.
     * 
     * @param logP the PDF of the distribution from which a sample is drawn.
     * @return one sample.
     */
    public static double sliceSampling(DoubleFunction<Double> logP, double initX) {
        double stepSize = 0.20;
        int numSliceSampling = 100;
        
//        for (int i = 0; i < 100; i++)
//            System.err.print(String.format("%.3f\t", logP.apply(i/100.0)));
//        System.err.println();
        
        Random random = new Random();
        int[] hist = new int[10];  // Histogram of samples
        double[] newXs = new double[numSliceSampling];
        for (int i = 0; i < numSliceSampling; i++) {
            double x;
            if (i == 0) x = initX;
            else x = newXs[i-1];
            
            double logY = logP.apply(x);
            double newLogY = logY + Math.log(random.nextDouble());
            
            double cut = stepSize * random.nextDouble();
            double start = Math.max(x - cut, 0);
            double end = Math.min(x - cut + stepSize, 1);
            while (start > 0 && logP.apply(start) > newLogY) {
                start = Math.max(start - stepSize, 0);
            }
            while (end < 1 && logP.apply(end) > newLogY) {
                end = Math.min(end + stepSize, 1);
            }
       
            while (true) {
                newXs[i] = start + (end - start) * random.nextDouble();
                if (logP.apply(newXs[i]) >= newLogY) break;
                if (newXs[i] < x) start = newXs[i];
                else end = newXs[i];
            }
//            System.err.println(start + " : " + end + " : " + newXs[i]);
            hist[(int)(newXs[i] / 0.1)]++;
        }
//        for (int i = 0; i < 10; i++) System.err.print(String.format("%d ", hist[i]));
//        System.err.println();
        return newXs[random.nextInt(numSliceSampling)];
    }

    /**
     * Returns the max value of the given array. 
     *
     * @param v the double array from which the max value is returned.
     * @return the max value of the given array.
     * @throws Exception if the array is empty.
     */
    public static double max(double[] v) throws Exception {
        return Arrays.stream(v).max().orElseThrow(
                () -> new RuntimeException("Array is empty"));
    }

    /**
     * Returns a value drawn from the categorical distribution with
     * the given array as its parameter. 
     *
     * @param probs the parameter of the categorical distribution.
     * @return the sampled index.
     */
    public static int sampleIndex(double[] probs) throws Exception {
        double[] cumulProbs = probs.clone();
        Arrays.parallelPrefix(cumulProbs, (x,y) -> x+y);
        double rand = Math.random() * cumulProbs[cumulProbs.length-1];
        return IntStream.range(0, cumulProbs.length)
                        .filter(i -> cumulProbs[i] >= rand)
                        .findFirst()
                        .orElseThrow(() -> new Exception("Sampling failed"));
    }

    /**
     * Generates output files. 
     *
     * @param iter the iteration number of the results to be output.
     */
    public static void genOutFiles(int iter) throws Exception {
        String outPathPrefix = outDir + "/I" + iter;
        saveModel(outPathPrefix);
        
        // Pi-State
        CSVPrinter outPiS = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-PiS.csv"), CSVFormat.EXCEL);
        outPiS.print("");
        for (int s = 0; s < numStates; s++) outPiS.print("S"+s);
        outPiS.println();

        outPiS.print("Init");
        for (int s2 = 0; s2 < numStates; s2++)
            outPiS.print((N_0S.getValue(0, s2) + sGamma) / (N_0S.getRowSum(0) + sumSGamma));
        outPiS.println();

        for (int s1 = 0; s1 < numStates; s1++) {
            outPiS.print("S"+s1);
            for (int s2 = 0; s2 < numStates; s2++) {
                outPiS.print((N_SS.getValue(s1, s2) + sGamma) / (N_SS.getRowSum(s1) + sumSGamma));
            }
            outPiS.println();
        }

        outPiS.flush();
        outPiS.close();


        // ThetaF
        CSVPrinter outFTheta = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-ThetaF.csv"), CSVFormat.EXCEL);
        outFTheta.print("");
        for (int t = 0; t < numFTopics; t++) outFTheta.print("FT"+t);
        outFTheta.println();

        for (int s = 0; s < numStates; s++) {
            outFTheta.print("S"+s);
            for (int t = 0; t < numFTopics; t++)
                outFTheta.print((N_SF.getValue(s,t) + fAlpha) / (N_SF.getRowSum(s) + sumFAlpha));
            outFTheta.println();
        }
        outFTheta.flush();
        outFTheta.close();

        // ThetaB
        CSVPrinter outBTheta = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-ThetaB.csv"), CSVFormat.EXCEL);
        for (int t = 0; t < numBTopics; t++) outBTheta.print("BT"+t);
        outBTheta.println();

        for (int t = 0; t < numBTopics; t++)
            outBTheta.print((N_B.getValue(0,t) + bAlpha) / (N_B.getRowSum(0) + sumBAlpha));
        outBTheta.println();

        outBTheta.flush();
        outBTheta.close();


        // DThetaF
        CSVPrinter outDThetaF = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-DThetaF.csv"), CSVFormat.EXCEL);
        outDThetaF.print("SeqId");
        outDThetaF.print("InstNo");
        for (int t = 0; t < numFTopics; t++) outDThetaF.print("FT"+t);
        outDThetaF.println();

        for (String seqKey : rawSeqs.keySet()) {
            TreeMap<Integer, RawInstance> dataSeq = rawSeqs.get(seqKey);
            Sequence seqVal = seqs.get(seqKey);
            for (int instNo : dataSeq.keySet()) {
                outDThetaF.print(seqKey);
                outDThetaF.print(instNo);

                Instance inst = null;
                if (seqVal != null) inst = seqVal.instances.get(instNo);
                if (inst != null) {
                    DoubleMatrix prob = new DoubleMatrix(1, numFTopics);
                    for (Sentence sentence : inst.sentences)
                        prob.incValue(0, sentence.fTopic);
                    for (int t = 0; t < numFTopics; t++) 
                        outDThetaF.print((prob.getValue(0, t) + fAlpha)
                                          / (prob.getRowSum(0) + sumFAlpha));
                }
                outDThetaF.println();
            }
        }
        outDThetaF.flush();
        outDThetaF.close();


        // PhiF
        CSVPrinter outPhiF = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-PhiF.csv"), CSVFormat.EXCEL);
        outPhiF.print("");
        for (int t = 0; t < numFTopics; t++) outPhiF.print("FT"+t);
        outPhiF.println();

        for (int w = 0; w < numWords; w++) {
            outPhiF.print(wordList.get(w));
            for (int t = 0; t < numFTopics; t++)
                outPhiF.print((N_FW.getValue(t,w) + beta) / (N_FW.getRowSum(t) + sumBeta));
            outPhiF.println();
        }
        outPhiF.flush();
        outPhiF.close();


        // PhiB
        CSVPrinter outPhiB = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-PhiB.csv"), CSVFormat.EXCEL);
        outPhiB.print("");
        if (useDomain) { 
            for (int t = 0; t < numBTopics; t++) 
                outPhiB.print(domainList.get(t)); 
        } else { 
            for (int t = 0; t < numBTopics; t++) 
                outPhiB.print("BT"+t); 
        }
        outPhiB.println();

        for (int w = 0; w < numWords; w++) {
            outPhiB.print(wordList.get(w));
            for (int t = 0; t < numBTopics; t++)
                outPhiB.print((N_BW.getValue(t,w) + beta) / (N_BW.getRowSum(t) + sumBeta));
            outPhiB.println();
        }
        outPhiB.flush();
        outPhiB.close();


        // Top words for each foreground topic
        CSVPrinter outProbWordsF = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-ProbWordsF.csv"), CSVFormat.EXCEL);
        for (int t = 0; t < numFTopics; t++) outProbWordsF.print("FT"+t);
        outProbWordsF.println();

        IntegerMatrix P_FW = N_FW.getSortedIndexMatrix(1, Math.min(numProbWords, numWords));
        for (int p = 0; p < Math.min(numProbWords, numWords); p++) {
            for (int t = 0; t < numFTopics; t++) {
                int w = P_FW.getValue(t,p);
                if (N_FW.getValue(t,w) < 1) outProbWordsF.print("");
                else outProbWordsF.print(String.format("%s (%.3f)", wordList.get(w), 
                        (N_FW.getValue(t,w) + beta) / (N_FW.getRowSum(t) + sumBeta)));
            }
            outProbWordsF.println();
        }
        outProbWordsF.flush();
        outProbWordsF.close();


        // Top words for each background topic
        CSVPrinter outProbWordsB = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-ProbWordsB.csv"), CSVFormat.EXCEL);
        if (useDomain) { 
            for (int t = 0; t < numBTopics; t++) 
                outProbWordsB.print(domainList.get(t)); 
        } else { 
            for (int t = 0; t < numBTopics; t++) 
                outProbWordsB.print("BT"+t); 
        }
        outProbWordsB.println();

        IntegerMatrix P_BW = N_BW.getSortedIndexMatrix(1, Math.min(numProbWords, numWords));
        for (int p = 0; p < Math.min(numProbWords, numWords); p++) {
            for (int t = 0; t < numBTopics; t++) {
                int w = P_BW.getValue(t,p);
                if (N_BW.getValue(t,w) < 1) outProbWordsB.print("");
                else outProbWordsB.print(String.format("%s (%.3f)", wordList.get(w), 
                        (N_BW.getValue(t,w) + beta) / (N_BW.getRowSum(t) + sumBeta)));
            }
            outProbWordsB.println();
        }
        outProbWordsB.flush();
        outProbWordsB.close();


        // Pi-Author
        CSVPrinter outPiA = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-PiA.csv"), CSVFormat.EXCEL);
        outPiA.print("SeqId");
        outPiA.print("Author");
        for (int s = 0; s < numStates; s++) outPiA.print("S"+s);
        outPiA.println();

        for (String seqKey : seqs.keySet()) {
            DoubleMatrix N_AS = N_QAS.get(seqKey);

            for (int a = 0; a < N_AS.getNumRows(); a++) {
                outPiA.print(seqKey);
                outPiA.print(seqAuthorList.get(seqKey).get(a));
                for (int s = 0; s < numStates; s++) {
                    outPiA.print((N_AS.getValue(a,s)+aGamma) / (N_AS.getRowSum(a)+sumAGamma));
                }
                outPiA.println();
            }
        }
        outPiA.flush();
        outPiA.close();


        // Instance-state/BT assignment
        printInstAssign(iter);


        // Instance(sentence)-state assignment
        CSVPrinter outSentAssign = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-InstSentAssign.csv"), CSVFormat.EXCEL);
        outSentAssign.printRecord("SeqId", "InstNo", "Author", "Sentence", "TaggedText", "State", 
                                                                            "FTopic", "BTopic");

        for (String seqKey : rawSeqs.keySet()) {
            TreeMap<Integer, RawInstance> rawSeq = rawSeqs.get(seqKey);
            Sequence seqVal = seqs.get(seqKey);
            for (int instNo : rawSeq.keySet()) {
                Instance inst = null;
                if (seqVal != null) inst = seqVal.instances.get(instNo);
                if (inst != null) {
                    for (int s = 0; s < inst.sentences.size(); s++) {
                        Sentence sentence = inst.sentences.get(s);
                        StringBuffer taggedText = new StringBuffer();
                        for (Word word : sentence.words) {
                            if (word.level==1) taggedText.append(
                                    "F"+sentence.fTopic+":" + wordList.get(word.id) + " ");
                            else taggedText.append(
                                    "B"+seqVal.bTopic+":" + wordList.get(word.id) + " ");
                        }
                        outSentAssign.printRecord(seqKey, instNo, rawSeq.get(instNo).author, s, 
                                taggedText.toString().trim(), inst.state, sentence.fTopic, 
                                seqVal.bTopic+(useDomain ? " ("+domainList.get(seqVal.bTopic)+")" : ""));
                    }
                }
                else outSentAssign.printRecord(seqKey, instNo, rawSeq.get(instNo).author, "", 
                                                        rawSeq.get(instNo).text, "", "", "");
            }
        }
        outSentAssign.flush();
        outSentAssign.close();

        // Log-likelihood
        if (logL.size() > 0) {
            CSVPrinter outLogL = new CSVPrinter(new PrintFileWriter(
                    outPathPrefix+"-LogL.csv"), CSVFormat.EXCEL);
            outLogL.printRecord("Iter","LogL");
            for (Map.Entry<Integer,Double> entry : logL.entrySet()) {
                outLogL.printRecord(entry.getKey(), entry.getValue());
            }
            outLogL.flush();
            outLogL.close();
        }
    }
    
    
    public static void printInstAssign(int iter) throws Exception {
        String outPathPrefix = outDir + "/I" + iter;
        
        CSVPrinter outInstAssign = new CSVPrinter(new PrintFileWriter(
                outPathPrefix+"-InstAssign.csv"), CSVFormat.EXCEL);

        outInstAssign.printRecord("SeqId", "InstNo", "Author", "Label", 
                                  "Text", "TaggedText", "State", "BTopic");

        for (String seqKey : rawSeqs.keySet()) {
            TreeMap<Integer, RawInstance> rawSeq = rawSeqs.get(seqKey);
            Sequence seqVal = seqs.get(seqKey);
            for (int instNo : rawSeq.keySet()) {
                Instance inst = null;
                if (seqVal != null) inst = seqVal.instances.get(instNo);
                if (inst != null) {
                    StringBuffer taggedText = new StringBuffer();
                    for (Sentence sentence : inst.sentences) {
                        for (Word word : sentence.words) {
                            if (word.level==1) 
                                taggedText.append(
                                    "F"+sentence.fTopic+":" + wordList.get(word.id) + " ");
                            else 
                                taggedText.append(
                                    "B"+seqVal.bTopic+":" + wordList.get(word.id) + " ");
                        }
                    }
                    outInstAssign.printRecord(seqKey, instNo, rawSeq.get(instNo).author, 
                            (labelIsGiven ? rawSeq.get(instNo).label : ""), rawSeq.get(instNo).text, 
                            taggedText.toString().trim(), inst.state, 
                            seqVal.bTopic+(useDomain ? " ("+domainList.get(seqVal.bTopic)+")" : ""));
                } else { 
                    outInstAssign.printRecord(seqKey, instNo, rawSeq.get(instNo).author, 
                            (labelIsGiven ? rawSeq.get(instNo).label : ""), rawSeq.get(instNo).text, "", "", "");
                }
            }
        }
        outInstAssign.flush();
        outInstAssign.close();
    }

    /**
     * Calculates and returns the log-likelihood of the data. 
     *
     * @return the log-likelihood of the data.
     */
    public static double logLikelihood() {
        DoubleMatrix P_SS = N_SS.copy();
        for (int r = 0; r < P_SS.getNumRows(); r++) {
            double denom = P_SS.getRowSum(r) + (numStates)*sGamma;
            for (int c = 0; c < P_SS.getNumColumns(); c++)
                P_SS.setValue(r,c, (P_SS.getValue(r,c)+sGamma) / denom);
        }
        DoubleMatrix P_0S = N_0S.copy();
        for (int r = 0; r < P_0S.getNumRows(); r++) {
            double denom = P_0S.getRowSum(r) + numStates*sGamma;
            for (int c = 0; c < P_0S.getNumColumns(); c++)
                P_0S.setValue(r,c, (P_0S.getValue(r,c)+sGamma) / denom);
        }
        DoubleMatrix P_FW = N_FW.copy();
        for (int r = 0; r < P_FW.getNumRows(); r++) {
            double denom = P_FW.getRowSum(r) + numWords*beta;
            for (int c = 0; c < P_FW.getNumColumns(); c++)
                P_FW.setValue(r,c, (P_FW.getValue(r,c)+beta) / denom);
        }
        DoubleMatrix P_BW = N_BW.copy();
        for (int r = 0; r < P_BW.getNumRows(); r++) {
            double denom = P_BW.getRowSum(r) + numWords*beta;
            for (int c = 0; c < P_BW.getNumColumns(); c++)
                P_BW.setValue(r,c, (P_BW.getValue(r,c)+beta) / denom);
        }
        DoubleMatrix P_SF = N_SF.copy();
        for (int r = 0; r < P_SF.getNumRows(); r++) {
            double denom = P_SF.getRowSum(r) + numFTopics*fAlpha;
            for (int c = 0; c < P_SF.getNumColumns(); c++)
                P_SF.setValue(r,c, (P_SF.getValue(r,c)+fAlpha) / denom);
        }
        DoubleMatrix P_B = N_B.copy();
        {
            double denom = P_B.getRowSum(0) + sumBAlpha;
            for (int c = 0; c < P_B.getNumColumns(); c++)
                P_B.setValue(0,c, (P_B.getValue(0,c)+bAlpha) / denom);
        }


        double logP = 0;
        for (String seqId : seqs.keySet()) {
            DoubleMatrix P_AS = N_QAS.get(seqId).copy();
            for (int r = 0; r < P_AS.getNumRows(); r++) {
                double denom = P_AS.getRowSum(r) + sumAGamma;
                for (int c = 0; c < numStates; c++)
                    P_AS.setValue(r,c, (P_AS.getValue(r,c)+aGamma) / denom);
            }

            Sequence seqVal = seqs.get(seqId);
            int bTopic = seqVal.bTopic; 

            // BTopic
            logP += Math.log(P_B.getValue(0, bTopic));

            Vector<Instance> seq = new Vector<Instance>(seqVal.instances.values());
            for (int d = 0; d < seq.size(); d++) {
                Instance inst = seq.get(d);
                int state = inst.state;

                // State transition
                if (inst.parent == null) logP += Math.log(nu * P_0S.getValue(0, state) 
                                            + (1-nu) * P_AS.getValue(inst.author, state));
                else logP += Math.log(nu * P_SS.getValue(inst.parent.state, state) 
                                            + (1-nu) * P_AS.getValue(inst.author, state));

                for (Sentence sentence : inst.sentences) {
                    // FTopic
                    logP += Math.log(P_SF.getValue(state, sentence.fTopic));

                    // Words
                    for (Word word : sentence.words) {
                        if (word.level == 1)
                            logP += Math.log(eta) + Math.log(P_FW.getValue(sentence.fTopic, word.id));
                        else
                            logP += Math.log(1-eta) + Math.log(P_BW.getValue(bTopic, word.id));
                    }
                }
            }
        }

        return logP;
    }
    
    /**
     * Saves {@link #wordCnt} to a file. 
     */  
    public static void saveWords() throws Exception {
        CSVPrinter outWordCount = new CSVPrinter(
                new PrintFileWriter(outDir+"/WordCount.csv"), CSVFormat.EXCEL);
        outWordCount.printRecord("Word","Count");
        for (String word : wordList)
            outWordCount.printRecord(word, wordCnt.get(word));
        outWordCount.flush();
        outWordCount.close();
    }
    
    /**
     * Saves {@link #domainList} to a file.
     */  
    public static void saveDomains() throws Exception {
        PrintFileWriter outDomains = new PrintFileWriter(outDir+"/Domains.txt");
        domainList.forEach(s -> outDomains.println(s));
        outDomains.close();
    }

    /**
     * Saves the current counter matrices to files.
     */  
    public static void saveModel(String outPathPrefix) throws Exception {
        N_SS.saveToCsvFile(outPathPrefix+"-N_SS.csv");
        N_0S.saveToCsvFile(outPathPrefix+"-N_0S.csv");
        N_FW.saveToCsvFile(outPathPrefix+"-N_FW.csv");
        N_BW.saveToCsvFile(outPathPrefix+"-N_BW.csv");
        N_SF.saveToCsvFile(outPathPrefix+"-N_SF.csv");
        N_B.saveToCsvFile(outPathPrefix+"-N_B.csv");
        // Save N_QAS
        // Just for inspection purposes. The saved variable is not restored.
        Files.createDirectories(Paths.get(outPathPrefix+"-N_AS"));
        int seqIdx = 0;
        for (DoubleMatrix N_AS : N_QAS.values()) {
            N_AS.saveToCsvFile(outPathPrefix+"-N_AS/"+String.format("%06d", seqIdx)+".csv");
            seqIdx++;
        }
        
        if (nuIsNotGiven) {
            PrintFileWriter out = new PrintFileWriter(outPathPrefix + "-Nu.txt");
            out.println(nu);
            out.close();
        }
        if (etaIsNotGiven) {
            PrintFileWriter out = new PrintFileWriter(outPathPrefix + "-Eta.txt");
            out.println(eta);
            out.close();
        }
    }

    /**
     * Restores pre-trained counter matrices from files.
     */
    public static void restoreModel() throws Exception {
        N_SS = DoubleMatrix.loadFromCsvFile(modelPathPrefix+"-N_SS.csv");
        N_0S = DoubleMatrix.loadFromCsvFile(modelPathPrefix+"-N_0S.csv");
        N_FW = DoubleMatrix.loadFromCsvFile(modelPathPrefix+"-N_FW.csv");
        N_BW = DoubleMatrix.loadFromCsvFile(modelPathPrefix+"-N_BW.csv");
        N_SF = DoubleMatrix.loadFromCsvFile(modelPathPrefix+"-N_SF.csv");
        N_B = DoubleMatrix.loadFromCsvFile(modelPathPrefix+"-N_B.csv");
        
        if (nuIsNotGiven) {
            BufferedFileReader in = new BufferedFileReader(modelPathPrefix + "-Nu.txt");
            nu = Double.valueOf(in.readLine());
            in.close();
        }
        if (etaIsNotGiven) {   
            BufferedFileReader in = new BufferedFileReader(modelPathPrefix + "-Eta.txt");
            eta = Double.valueOf(in.readLine());
            in.close();
        }
    }
    
    /**
     * Loads indexed words {@link #wordCnt}, {@link #wordIndex}, 
     * and {@link #wordList} from files.
     */
    public static void loadWords() throws Exception {
        // Load wordCount, wordIndex, and wordList
        CSVParser csvWordCount = new CSVParser(
                new BufferedFileReader(modelPathPrefix.replaceAll("I\\d+$", "WordCount.csv")),
                CSVFormat.EXCEL.withHeader());
        for (CSVRecord record : csvWordCount) {
            String word = record.get("Word");
            double cnt = Double.valueOf(record.get("Count"));
            wordCnt.increase(word, cnt);
            wordIndex.put(word, wordIndex.size());
            wordList.add(word);
        }
        csvWordCount.close();
        numWords = wordIndex.size();
    }
    
    /**
     * Loads saved domains to {@link #domainIndex} and {@link #domainList}.
     */
    public static void loadDomains() throws Exception {
        BufferedFileReader domainFile = 
                new BufferedFileReader(modelPathPrefix.replaceAll("I\\d+$", "Domains.txt"));
        while (domainFile.nextLine()) {
            String domain = domainFile.readLine();
            domainIndex.put(domain, domainIndex.size());
            domainList.add(domain);
        }
        domainFile.close();
        numBTopics = domainIndex.size();
    }

    /**
     * A raw instance before being indexed.
     */
    private static class RawInstance {
        String author = null;
        String label = null;
        String text = null;
        Vector<Vector<String>> sentences = new Vector<Vector<String>>();
    }

    /**
     * A sequence.
     */
    private static class Sequence {
        int bTopic = -1;
        TreeMap<Integer,Instance> instances = new TreeMap<Integer,Instance>();
    }

    /**
     * An instance.
     */
    private static class Instance {
        int state = -1;
        int author = -1;
        Instance parent = null;
        Vector<Instance> children = new Vector<Instance>();
        Vector<Sentence> sentences = new Vector<Sentence>();
    }

    /**
     * A sentence.
     */
    private static class Sentence {
        int fTopic = -1;
        Vector<Word> words = new Vector<Word>();
    }

    /**
     * A word.
     */
    private static class Word {
        int id = -1;
        int level = -1;
        public Word(int id) {
            this.id = id;
        }
    }
}