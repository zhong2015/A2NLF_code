package recommender.common;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

public class CommonRec {

    public static final int RMSE = 1;
    public static final int MAE = 2;

    public static String dataSetName;
    public static ArrayList<RTuple> trainDataSet =  null;
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
    public double cacheTotalTime = 0;
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
    public double[][] Q;

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

    public CommonRec() {
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
        Q = new double[itemMaxID + 1][featureDimension];

        Gamma_X = new double[userMaxID + 1][featureDimension];
        Gamma_Y = new double[itemMaxID + 1][featureDimension];

        for (int u = 1; u <= userMaxID; u++) {
            for (int dim = 0; dim < featureDimension; dim++) {
                X[u][dim] = catchedUserXFeatureMatrix[u][dim];
                P[u][dim] = catchedUserXFeatureMatrix[u][dim];
                Gamma_X[u][dim] = 0;
            }
        }
        for (int i = 1; i <= itemMaxID; i++) {
            for (int dim = 0; dim < featureDimension; dim++) {
                Y[i][dim] = catchedItemYFeatureMatrix[i][dim];
                Q[i][dim] = catchedItemYFeatureMatrix[i][dim];
                Gamma_Y[i][dim] = 0;
            }
        }
    }

    /*
     * 生成初始的训练集、测试集以及统计各个结点的评分数目
     */
    public static void dataLoad(String trainFileName, String testFileName, String separator) throws IOException {

        //生成初始的训练集
        trainDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, trainFileName, trainDataSet);

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

    /*
     * 计算考虑线性偏差的预测值
     */
    public double getPrediction(int userID, int itemID) {
        double ratingHat = 0;
        ratingHat += dotMultiply(P[userID], Q[itemID]);
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

    public double testRMSE() {

        // 计算在测试集上的RMSE
        double sumRMSE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
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
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumMAE += Math.abs(actualRating - ratinghat);
            sumCount++;
        }
        double MAE = sumMAE / sumCount;
        return MAE;
    }

    public void outPutModelNonnega(int round_num, double curErr) throws IOException {

        FileWriter fw = new FileWriter(
                new File("./" + dataSetName + "_" + round_num + "_" + curErr + "_ModelNonnega.txt"), true);
        for(int u = 1; u <= userMaxID; u++) {
            for(int dim = 0; dim < featureDimension; dim++){
                fw.write(P[u][dim] + "\n");
                fw.flush();
            }
        }

        for(int i = 1; i <= itemMaxID; i++) {
            for(int dim = 0; dim < featureDimension; dim++){
                fw.write(Q[i][dim] + "\n");
                fw.flush();
            }
        }
        fw.close();
    }

    public void printNegativeFeature() {

        System.out.println("************************** negative feature:**************************");
        for (int u = 1; u <= userMaxID; u++)
            for (int dim = 0; dim < featureDimension; dim++)
                if (P[u][dim] < 0)
                    System.out.println(P[u][dim]);

        for (int i = 1; i <= itemMaxID; i++)
            for (int dim = 0; dim < featureDimension; dim++)
                if (Q[i][dim] < 0)
                    System.out.println(Q[i][dim]);

        System.out.println("***************************************************************************");

    }
}
