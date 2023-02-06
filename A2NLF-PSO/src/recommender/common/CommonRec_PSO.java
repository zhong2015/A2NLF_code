package recommender.common;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

public class CommonRec_PSO {

    public static final int RMSE = 1;
    public static final int MAE = 2;

    public static String dataSetName;
    public static ArrayList<RTuple> trainDataSet =  null;
    public static ArrayList<RTuple> valDataSet =  null;
    public static ArrayList<RTuple> testDataSet =  null;

    public static int userMaxID = 0;
    public static int itemMaxID = 0;

    public static double userRSetSize[];
    public static double itemRSetSize[];

    public static double lambda = 0.005; // 增广项参数
    public static int maxRound = 500; // 最多训练轮数
    public static int featureDimension = 0; // 特征维数
    public static double minGap = 0;
    public static int delayCount = 100;

    public double  min_Error = 1e10; // 最小误差值
    public double cacheMinTotalTime = 0;
    public double minTotalTime = 0;
    public int min_Round = 0; // 记录达到最优结果时的最小迭代次数
    public int total_Round = 0;
    public double total_Time = 0;

    public static double[][] catchedUserXFeatureMatrix;
    public static double[][] catchedItemYFeatureMatrix;
    public static int mappingScale = 1000;
    public static double featureInitMax = 0.004;
    public static double featureInitScale = 0.004;

    public double[][] P;
    public double[][] Min_P;
    public double[][] Q;
    public double[][] Min_Q;

    // 存储特征值的路径
    public static String userXFeatureSaveDir;
    public static String itemYFeatureSaveDir;

    // ANLF所需的额外参数
    public static double eta = 1;// 学习率参数
    // 对应P，X和对应拉格朗日乘子
    public static double[][] X,Gamma_X;
    // 对应Q，Y和对应拉格朗日乘子
    public static double[][] Y,Gamma_Y;
    // 进行更新时的缓存矩阵
    public static double[] X_U, X_D, X_C;
    public static double[] Y_U, Y_D, Y_C;
    
    //PSO
    public double  last_Error = 100; // 测试集的最小RMSE值
    
    public static double lambdaMax = 2, lambdaMin = 0.2;
    public static double etaMax = 2, etaMin = 1;
    
    public static String ParticlesSaveDir;
    public static String VSaveDir;
    
    public static double[][] particles;
    public static double[][] V;
    public static double[][] pbest;
    public static double[] gbest;
    
    public static int hyperNum = 2;
    public static int swarmNum = 10;
    public static double c1 = 2;
    public static double c2 = 2;
    public static double w = 0.729;
    public static double r1 ;
    public static double r2 ;
    public static double Vmax =1;
    public static double Vmin =-1;
    public static double[] FitnessRMSE;
    public static double[] FitnessRMSEpbest;
    public static double FitnessRMSEgbest;
    public static double[] FitnessMAE;
    public static double[] FitnessMAEpbest;
    public static double FitnessMAEgbest;
    public static int bestSwarm = 0;
    
    //FF
    public static double[] new_Fitness_U;
    public static double[] new_Fitness_D;
    public static double[] nFitnesspbest;
    public static double nFitnessgbest;
    public static int bestSwarm_nFitness = 0;
    
    public double train_last_valRMSE = 0;
    public double train_last_valMAE = 0;
    
    //cg
    public double FitnessRMSEcbest;
    public double FitnessMAEcbest;
    
    //cc
    public double last_FitnessRMSEcbest;
    public double last_FitnessMAEcbest;

    public ArrayList<ArrayList<Double>> lambda_particles;
    public ArrayList<ArrayList<Double>> eta_particles;

    public CommonRec_PSO() {
        this.initInstanceFeatures();
    }

    /*
     * 初始化实例特征矩阵
     */
    public void initInstanceFeatures() {
        // 加1是为了在序号上与ID保持一致
        X = new double[userMaxID + 1][featureDimension];
        Y = new double[itemMaxID + 1][featureDimension];

        P = new double[userMaxID + 1][featureDimension];
        Min_P = new double[userMaxID + 1][featureDimension];
        Q = new double[itemMaxID + 1][featureDimension];
        Min_Q = new double[itemMaxID + 1][featureDimension];

        Gamma_X = new double[userMaxID + 1][featureDimension];
        Gamma_Y = new double[itemMaxID + 1][featureDimension];

        for (int u = 1; u <= userMaxID; u++) {
            for (int dim = 0; dim < featureDimension; dim++) {
                X[u][dim] = catchedUserXFeatureMatrix[u][dim];
                P[u][dim] = catchedUserXFeatureMatrix[u][dim];
                Min_P[u][dim] = catchedUserXFeatureMatrix[u][dim];
                Gamma_X[u][dim] = 0;
            }
        }
        for (int i = 1; i <= itemMaxID; i++) {
            for (int dim = 0; dim < featureDimension; dim++) {
                Y[i][dim] = catchedItemYFeatureMatrix[i][dim];
                Q[i][dim] = catchedItemYFeatureMatrix[i][dim];
                Min_Q[i][dim] = catchedItemYFeatureMatrix[i][dim];
                Gamma_Y[i][dim] = 0;
            }
        }
    }

    /*
     * 生成初始的训练集、测试集以及统计各个结点的评分数目
     */
    public static void dataLoad(String trainFileName, String valFileName, String testFileName, String separator) throws IOException {

        //生成初始的训练集
        trainDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, trainFileName, trainDataSet);
        
        //生成初始的验证集
        valDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, valFileName, valDataSet);

        //生成初始的测试集
        testDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, testFileName, testDataSet);

        initRatingSetSize();
    }

    /*
     * 数据集生成器
     */
    public static void dataSetGenerator(String separator, String fileName, ArrayList<RTuple> dataSet) throws IOException {

        File fileSource = new File(fileName);
        BufferedReader in = new BufferedReader(new FileReader(fileSource));

        String line;
        while (((line = in.readLine()) != null)){
            StringTokenizer st = new StringTokenizer(line, separator);
            String personID = null;
            if (st.hasMoreTokens())
                personID = st.nextToken();
            String movieID = null;
            if (st.hasMoreTokens())
                movieID = st.nextToken();
            String personRating = null;
            if (st.hasMoreTokens())
                personRating = st.nextToken();
            int iUserID = Integer.valueOf(personID);
            int iItemID = Integer.valueOf(movieID);

            // 记录下最大的itemid和userid；因为itemid和userid是连续的，所以最大的itemid和userid也代表了各自的数目
            userMaxID = (userMaxID > iUserID) ? userMaxID : iUserID;
            itemMaxID = (itemMaxID > iItemID) ? itemMaxID : iItemID;
            double dRating = Double.valueOf(personRating);

            RTuple tempR = new RTuple();
            tempR.userID = iUserID;
            tempR.itemID = iItemID;
            tempR.rating = dRating;
            dataSet.add(tempR);
        }
        in.close();
    }

    /*
     * 统计各个结点的评分数目
     */
    public static void initRatingSetSize() {
        userRSetSize = new double[userMaxID + 1];
        itemRSetSize = new double[itemMaxID + 1];

        for (int u = 1; u <= userMaxID; u++) {
            userRSetSize[u] = 0;
        }
        for (int i = 1; i <= itemMaxID; i++) {
            itemRSetSize[i] = 0;
        }

        for (RTuple tempRating : trainDataSet) {
            userRSetSize[tempRating.userID] += 1;
            itemRSetSize[tempRating.itemID] += 1;
        }
    }

    /*
     * 声明辅助矩阵，并用随机数进行初始化
     */
    public static void initStaticFeatures() throws IOException {

        // 加1是为了在序号上与ID保持一致
        catchedUserXFeatureMatrix = new double[userMaxID + 1][featureDimension];
        catchedItemYFeatureMatrix = new double[itemMaxID + 1][featureDimension];

        userXFeatureSaveDir = userXFeatureSaveDir + featureDimension + ".txt";
        itemYFeatureSaveDir = itemYFeatureSaveDir + featureDimension + ".txt";

        File userXfeatureFile = new File(userXFeatureSaveDir);        // new File(".") 表示用当前路径 生成一个File实例!!!并不是表达创建一个 . 文件
        File itemYfeatureFile = new File(itemYFeatureSaveDir);

        if(userXfeatureFile.exists() && itemYfeatureFile.exists()) {
            System.out.println("准备读取指定初始值...");
            readFeatures(catchedUserXFeatureMatrix, userXFeatureSaveDir);  // 读取特征矩阵
            readFeatures(catchedItemYFeatureMatrix, itemYFeatureSaveDir);
            System.out.println("读取完毕！！！");
        }else{
            System.out.println("准备生成随机初始值...");
            // 初始化特征矩阵,采用随机值,从而形成一个K阶逼近
            Random random = new Random(System.currentTimeMillis());

            for(int u = 1; u <= userMaxID; u++){
                // 特征矩阵的初始值在(0,0.004]
                for (int dim = 0; dim < featureDimension; dim++) {
                    int temp = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    catchedUserXFeatureMatrix[u][dim] = featureInitMax - featureInitScale * temp / mappingScale;
                }
            }

            for (int i = 1; i <= itemMaxID; i++) {

                // 特征矩阵的初始值在(0,0.004]
                for (int dim = 0; dim < featureDimension; dim++) {
                    int temp = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    catchedItemYFeatureMatrix[i][dim] = featureInitMax - featureInitScale * temp / mappingScale;
                }
            }

            // 写入文件
            writeFeatures(catchedUserXFeatureMatrix, userMaxID, userXFeatureSaveDir);
            writeFeatures(catchedItemYFeatureMatrix, itemMaxID, itemYFeatureSaveDir);
            System.out.println("写入随机初始值完毕！！！");
        }

        // 声明辅助矩阵
        initAuxArray();
    }

    private static void writeFeatures(double[][] catchedFeatureMatrix, int maxID, String featureSaveDir) throws IOException {

        FileWriter fw = new FileWriter(featureSaveDir);

        for(int i = 1; i <= maxID; i++) {
            for(int k = 0; k < featureDimension; k++) {
                fw.write(catchedFeatureMatrix[i][k] + "::");
            }
            fw.write("\n");
        }
        fw.flush();
        fw.close();
    }

    private static void readFeatures(double[][] catchedFeatureMatrix, String featureSaveDir) throws IOException {

        BufferedReader in = new BufferedReader(new FileReader(featureSaveDir));
        String line;  // 一行数据
        int i = 1;    // 行标
        while((line = in.readLine()) != null){
            String [] temp = line.split("::"); // 各数字之间用"::"间隔
            for(int k = 0; k < featureDimension; k++) {
                catchedFeatureMatrix[i][k] = Double.valueOf(temp[k]);
            }
            i++;
        }
        in.close();
    }

    /*
     * 声明辅助矩阵
     */
    public static void initAuxArray() {

        // 加1是为了在序号上与ID保持一致
        X_U = new double[userMaxID + 1];
        X_D = new double[userMaxID + 1];
        X_C = new double[userMaxID + 1];
        Y_U = new double[itemMaxID + 1];
        Y_D = new double[itemMaxID + 1];
        Y_C = new double[itemMaxID + 1];
    }

    /*
     * 将辅助矩阵的元素置为0
     */
    public static void resetAuxArray() {
        for (int u = 1; u <= userMaxID; u++) {
            X_U[u] = 0;
            X_D[u] = 0;
        }
        for (int i = 1; i <= itemMaxID; i++) {
            Y_U[i] = 0;
            Y_D[i] = 0;
        }
    }
    
    public static void initPSO() throws IOException {
        particles = new double[swarmNum][hyperNum];
        V = new double[swarmNum][hyperNum];
        pbest = new double[swarmNum][hyperNum];
        gbest = new double[hyperNum];
        double minVal = Math.min((lambdaMax - lambdaMin), (etaMax - etaMin));
        Vmax = 0.2 * minVal;
        Vmin=-Vmax;

        ParticlesSaveDir = ParticlesSaveDir + featureDimension + ".txt";
        VSaveDir = VSaveDir + featureDimension + ".txt";

        File ParticlesFile = new File(ParticlesSaveDir);
        File VFile = new File(VSaveDir);

        if(ParticlesFile.exists() && VFile.exists()){
            System.out.println("准备读取PSO_init指定初始值...");
            readPSO(particles, ParticlesSaveDir);
            readPSO(V, VSaveDir);
            System.out.println("读取完毕！！！");
            for (int q = 0; q < swarmNum; q++) {
                for (int i = 0; i < hyperNum; i++){
                    gbest[i] = 100;
                    pbest[q][i] = particles[q][i];
                }
            }
        }else{
            System.out.println("准备生成PSO_init随机初始值...");
            Random random = new Random(System.currentTimeMillis());
            for (int q = 0; q < swarmNum; q++) {
                // 给lambda的q个粒子赋初值, 取值范围：[lambdaMin, lambdaMax + minGap)
                double temp1 = lambdaMin + random.nextDouble()*(lambdaMax - lambdaMin + minGap);
                particles[q][0] = temp1;
                // 给eta的q个粒子赋初值
                double temp2 = etaMin + random.nextDouble()*(etaMax - etaMin + minGap);
                particles[q][1] = temp2;
                for (int i = 0; i < hyperNum; i++){
                    gbest[i] = 100;
                    pbest[q][i] = particles[q][i];
                    V[q][i] = random.nextDouble() * Vmax;
                }
            }
            writePSO(particles, ParticlesSaveDir);
            writePSO(V, VSaveDir);
            System.out.println("写入随机初始值完毕！！！");
        }
    }
    
    private static void writePSO(double[][] catchedPSOMatrix, String PSOSaveDir) throws IOException {

        FileWriter fw = new FileWriter(PSOSaveDir);

        for(int i = 0; i < swarmNum; i++) {
            for(int j = 0; j < hyperNum; j++) {
                fw.write(catchedPSOMatrix[i][j] + "::");
            }
            fw.write("\n");
        }
        fw.flush();
        fw.close();
    }

    private static void readPSO(double[][] catchedPSOMatrix, String PSOSaveDir) throws IOException {

        BufferedReader in = new BufferedReader(new FileReader(PSOSaveDir));
        String line;  // 一行数据
        int i = 0;    // 行标
        while((line = in.readLine()) != null){
            String [] temp = line.split("::"); // 各数字之间用"::"间隔
            for(int j = 0; j < hyperNum; j++) {
                catchedPSOMatrix[i][j] = Double.valueOf(temp[j]);
            }
            i++;
        }
        in.close();
    }
    
    public static void initFitness(int metrics){
    	
    	if(metrics == CommonRec_PSO.RMSE){
    		FitnessRMSE = new double[swarmNum];
            new_Fitness_U = new double[swarmNum];
            new_Fitness_D = new double[swarmNum];
            FitnessRMSEpbest = new double[swarmNum];
            nFitnesspbest = new double[swarmNum];
            FitnessRMSEgbest = 100;
            nFitnessgbest = 0;
            for (int j = 0; j < swarmNum; j++){
            	FitnessRMSEpbest[j] = 100;
            	nFitnesspbest[j] = 0;
            	new_Fitness_U[j] = 0;
            	new_Fitness_D[j] = 0;
            	FitnessRMSE[j] = 100;
            }
    	}else{
    		FitnessMAE = new double[swarmNum];
            new_Fitness_U = new double[swarmNum];
            new_Fitness_D = new double[swarmNum];
            FitnessMAEpbest = new double[swarmNum];
            nFitnesspbest = new double[swarmNum];
            FitnessMAEgbest = 100;
            nFitnessgbest = 0;
            for (int j = 0; j < swarmNum; j++){
            	FitnessMAEpbest[j] = 100;
            	nFitnesspbest[j] = 0;
            	new_Fitness_U[j] = 0;
            	new_Fitness_D[j] = 0;
            	FitnessMAE[j] = 100;
            }
    	}	
    }
    
    public double computingFitness_cc(double [] val_res, double last_val_res, int q, String flag) {
    	
    	if(flag == "D"){
    		double F_D = FitnessRMSEcbest - last_FitnessRMSEcbest;
    		return F_D;
    	}else{
    		double F_U;
        	if(q == 0){
        		F_U = val_res[q] - last_val_res;
        	}else{
        		F_U = val_res[q] - val_res[q - 1];
        	}
        	return F_U;
    	}
    }

    public void updateBestnFitness(double Fitness_U_q, double Fitness_D_q, int q) {
    	
    	if(Double.isNaN(Fitness_U_q) || Double.isInfinite(Fitness_U_q)){
    		Fitness_U_q = 100;
    	}
    	
    	if(Double.isNaN(Fitness_D_q) || Double.isInfinite(Fitness_D_q)){
    		Fitness_D_q = 100;
    	}
    	
    	if(Fitness_U_q <0 && Fitness_D_q < 0){
    		
    		double nFitness = Fitness_U_q / Fitness_D_q;
    		if(nFitness > nFitnesspbest[q]){
        		nFitnesspbest[q] = nFitness;
        		pbest[q] = particles[q];
        	}
    		
        	if(nFitness > nFitnessgbest){
        		nFitnessgbest = nFitness;
        		bestSwarm_nFitness = q;
        		gbest = particles[bestSwarm_nFitness];
        	}
    	}
    }
    
    public void updateBestRMSE(double FitnessRMSE_q, int q) {
    	if(Double.isNaN(FitnessRMSE_q) || Double.isInfinite(FitnessRMSE_q)){
    		FitnessRMSEpbest[q] = 100;
    	}
    	
        if (FitnessRMSE_q < FitnessRMSEpbest[q]) {
            FitnessRMSEpbest[q] = FitnessRMSE_q;
        }

        if (FitnessRMSE_q < FitnessRMSEgbest) {
            FitnessRMSEgbest = FitnessRMSE_q;
            cacheMin();
        }
    }
    
    public void updateBestMAE(double FitnessMAE_q, int q) {
    	if(Double.isNaN(FitnessMAE_q) || Double.isInfinite(FitnessMAE_q)){
    		FitnessMAEpbest[q] = 100;
    	}
    	
        if (FitnessMAE_q < FitnessMAEpbest[q]) {
            FitnessMAEpbest[q] = FitnessMAE_q;
        }

        if (FitnessMAE_q < FitnessMAEgbest) {
            FitnessMAEgbest = FitnessMAE_q;
            cacheMin();
        }
    }
    
    public void cacheMin(){
    
    	for(int id = 1; id <= userMaxID; id++){
    		for(int dim = 0; dim < featureDimension; dim++){
    			Min_P[id][dim] = P[id][dim];
    		}
		}
    	for(int id = 1; id <= itemMaxID; id++){
    		for(int dim = 0; dim < featureDimension; dim++){
    			Min_Q[id][dim] = Q[id][dim];
    		}
		}
    }
    
    public void update_Particles_V(int q){
        
        for (int i = 0; i < hyperNum; i++){
        	V[q][i] = w * V[q][i] + c1 * r1 * (pbest[q][i] - particles[q][i]) + c2 * r2 * (gbest[i] - particles[q][i]);
            
            // 比最大值还大则赋值为最大值，比最小值还小则赋值为最小值
            if(V[q][i] > Vmax){
                V[q][i] = Vmax;
            }else if(V[q][i] < Vmin){
                V[q][i] = Vmin;
            }

            particles[q][i] = particles[q][i] + V[q][i];

            // 比上界还大则赋值为上界，比下界还小则赋值为下界
            if(particles[q][0] > lambdaMax){
                particles[q][0] = lambdaMax;
            }
            if(particles[q][0] < lambdaMin){
                particles[q][0] = lambdaMin;
            }
            if(particles[q][1] > etaMax){
                particles[q][1] = etaMax;
            }
            if(particles[q][1] < etaMin){
                particles[q][1] = etaMin;
            }
        }
    }

    /*
     * 计算考虑线性偏差的预测值
     */
    public double getPrediction(int userID, int itemID) {
        double ratingHat = 0;
        ratingHat += dotMultiply(P[userID], Q[itemID]);
        return ratingHat;
    }
    
    public double getMinPrediction(int userID, int itemID) {
        double ratingHat = 0;
        ratingHat += dotMultiply(Min_P[userID], Min_Q[itemID]);
        return ratingHat;
    }

    // 计算两个向量点乘
    public static double dotMultiply(double[] x, double[] y) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }
    
    public double valRMSE() {

        // 计算在测试集上的RMSE
        double sumRMSE = 0, sumCount = 0;
        for (RTuple testR : valDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumRMSE += Math.pow((actualRating - ratinghat), 2);
            sumCount++;
        }
        double RMSE = Math.sqrt(sumRMSE / sumCount);
        return RMSE;
    }

    public double valMAE() {
        // 计算在测试集上的MAE
        double sumMAE = 0, sumCount = 0;
        for (RTuple testR : valDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumMAE += Math.abs(actualRating - ratinghat);
            sumCount++;
        }
        double MAE = sumMAE / sumCount;
        return MAE;
    }
    
    public double testRMSE() {

        // 计算在测试集上的RMSE
        double sumRMSE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getMinPrediction(testR.userID, testR.itemID);
            sumRMSE += Math.pow((actualRating - ratinghat), 2);
            sumCount++;
        }
        double RMSE = Math.sqrt(sumRMSE / sumCount);
        return RMSE;
    }

    public double testMAE() {
        // 计算在测试集上的MAE
        double sumMAE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getMinPrediction(testR.userID, testR.itemID);
            sumMAE += Math.abs(actualRating - ratinghat);
            sumCount++;
        }
        double MAE = sumMAE / sumCount;
        return MAE;
    }
    
    public void printResults_v2(int metrics, int featureDimension) throws IOException {
    	FileWriter Y_fw;
    	FileWriter YHat_fw;
    	FileWriter P_fw;
    	FileWriter Q_fw;
    	if(metrics == CommonRec_PSO.RMSE){
    		Y_fw = new FileWriter(new File("./" + dataSetName + "_dim=" + featureDimension + "_RMSE_Y.txt"), true);
        	YHat_fw = new FileWriter(new File("./" + dataSetName  + "_dim=" + featureDimension + "_RMSE_YHat.txt"), true);
        	P_fw = new FileWriter(new File("./" + dataSetName  + "_dim=" + featureDimension + "_RMSE_A.txt"), true);
        	Q_fw = new FileWriter(new File("./" + dataSetName  + "_dim=" + featureDimension + "_RMSE_X.txt"), true);
    	}else{
    		Y_fw = new FileWriter(new File("./" + dataSetName + "_dim=" + featureDimension + "_MAE_Y.txt"), true);
        	YHat_fw = new FileWriter(new File("./" + dataSetName  + "_dim=" + featureDimension + "_MAE_YHat.txt"), true);
        	P_fw = new FileWriter(new File("./" + dataSetName  + "_dim=" + featureDimension + "_MAE_A.txt"), true);
        	Q_fw = new FileWriter(new File("./" + dataSetName  + "_dim=" + featureDimension + "_MAE_X.txt"), true);
    	}
    	
    	for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getMinPrediction(testR.userID, testR.itemID);
            Y_fw.write(testR.userID + "::" + testR.itemID + "::" + actualRating + "\n");
            YHat_fw.write(testR.userID + "::" + testR.itemID + "::" + ratinghat + "\n");
            Y_fw.flush();
            YHat_fw.flush();
        }
    	
    	for (int userID = 1; userID <= userMaxID; userID++){
    		for (int dim = 0; dim < featureDimension; dim++) {
    			P_fw.write(userID + "::" + dim + "::" + Min_P[userID][dim] + "\n");
    			P_fw.flush();
    		}
    	}
    	
    	for (int itemID = 1; itemID <= itemMaxID; itemID++){
    		for (int dim = 0; dim < featureDimension; dim++) {
    			Q_fw.write(itemID + "::" + dim + "::" + Min_Q[itemID][dim] + "\n");
    			Q_fw.flush();
    		}
    	}
    	
    	Y_fw.close();
		YHat_fw.close();
		P_fw.close();
		Q_fw.close();
    }

    public void initParticleList(){
        lambda_particles = new ArrayList<ArrayList<Double>>();
        eta_particles = new ArrayList<ArrayList<Double>>();

        for(int q = 0; q < swarmNum; q++){
            ArrayList<Double> temp1 = new ArrayList<Double>();
            lambda_particles.add(temp1);
            ArrayList<Double> temp2 = new ArrayList<Double>();
            eta_particles.add(temp2);
        }
    }

    public void recordParticles(){

        for(int q = 0; q < swarmNum; q++){
            double lambda = particles[q][0];
            lambda_particles.get(q).add(lambda);

            double eta = particles[q][1];
            eta_particles.get(q).add(eta);
        }
    }

    public void printParticles(int metrics) throws IOException {

        for(int q = 0; q < swarmNum; q++){
            FileWriter lambda_fw, eta_fw;
            if(metrics == CommonRec_PSO.RMSE){
                lambda_fw = new FileWriter(new File("./" + dataSetName  + "_RMSE_lambda_q=" + q + ".txt"), true);
                eta_fw = new FileWriter(new File("./" + dataSetName  + "_RMSE_eta_q=" + q + ".txt"), true);
            }else{
                lambda_fw = new FileWriter(new File("./" + dataSetName  + "_MAE_lambda_q=" + q + ".txt"), true);
                eta_fw = new FileWriter(new File("./" + dataSetName  + "_MAE_eta_q=" + q + ".txt"), true);
            }

            for(double alpha: lambda_particles.get(q)){
                lambda_fw.write(alpha + "\n");
                lambda_fw.flush();
            }
            lambda_fw.close();

            for(double beta: eta_particles.get(q)){
                eta_fw.write(beta + "\n");
                eta_fw.flush();
            }
            eta_fw.close();

        }
    }
}
