package recommender;

import recommender.common.CommonRec_PSO;
import recommender.common.RTuple;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class ANLF_PSO_cc extends CommonRec_PSO {

    public ANLF_PSO_cc(){
        super();
    }

    public static void main(String[] args) throws IOException {

        CommonRec_PSO.dataSetName = "Jester_200";
        String filePath = "E:\\workspace-jasonchung\\Selected_DS\\7-1-2\\";
        CommonRec_PSO.dataLoad(filePath + CommonRec_PSO.dataSetName + "_train.txt",filePath + CommonRec_PSO.dataSetName + "_val.txt", filePath + CommonRec_PSO.dataSetName + "_test.txt","::");
        System.out.println("Max User ID:\t" + CommonRec_PSO.userMaxID);
        System.out.println("Max Item ID:\t" + CommonRec_PSO.itemMaxID);
        System.out.println("训练集的容量："+ CommonRec_PSO.trainDataSet.size());
        System.out.println("验证集的容量："+ CommonRec_PSO.valDataSet.size());
        System.out.println("测试集的容量："+ CommonRec_PSO.testDataSet.size());

        // 设置公共参数
        CommonRec_PSO.maxRound = 100;
        CommonRec_PSO.minGap = 1e-5;
        CommonRec_PSO.delayCount = 10;

        for(int tempdim = 40; tempdim <= 40; tempdim += CommonRec_PSO.featureDimension){

            CommonRec_PSO.featureDimension = tempdim;
            CommonRec_PSO.userXFeatureSaveDir = "./savedLFs/"+ CommonRec_PSO.dataSetName +"/X";
            CommonRec_PSO.itemYFeatureSaveDir = "./savedLFs/"+ CommonRec_PSO.dataSetName +"/Y";
            CommonRec_PSO.ParticlesSaveDir = "./savedLFs/"+ CommonRec_PSO.dataSetName +"/Particles";
            CommonRec_PSO.VSaveDir = "./savedLFs/"+ CommonRec_PSO.dataSetName +"/V";

            // 初始化特征矩阵
            CommonRec_PSO.initStaticFeatures();
            CommonRec_PSO.initPSO();
            
//            experimenter(CommonRec_PSO.RMSE);
            experimenter(CommonRec_PSO.MAE);
        }
    }

    public static void experimenter(int metrics) throws IOException {
    	
    	CommonRec_PSO.initFitness(metrics);
    	
        long file_tMills = System.currentTimeMillis(); //用于给train函数打开在当前函数所创建的文件
        FileWriter fw;
        if(metrics == CommonRec_PSO.RMSE)
            fw = new FileWriter(new File("./" + CommonRec_PSO.dataSetName + "_RMSE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "dim=" + featureDimension + ".txt"), true);
        else
            fw = new FileWriter(new File("./" + CommonRec_PSO.dataSetName + "_MAE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "dim=" + featureDimension + ".txt"), true);

        String blankStr = "                          ";
        String starStr = "****************************************************************";
        
        // 打印标题项
        System.out.println("\n" + starStr);
        System.out.println(blankStr + "featureDimension——>" + CommonRec_PSO.featureDimension);
        System.out.println(blankStr + "swarmNum——>" + swarmNum);
        System.out.println(blankStr + "lambda——>" + lambdaMin + "-" + lambdaMax);
        System.out.println(blankStr + "eta——>" + etaMin + "-" + etaMax);
        System.out.println(blankStr + "minGap——>" + CommonRec_PSO.minGap);
        System.out.println(starStr);

        fw.write("\n" + starStr + "\n");
        fw.write(blankStr + "featureDimension——>" + CommonRec_PSO.featureDimension + "\n");
        fw.write(blankStr + "swarmNum——>" + swarmNum + "\n");
        fw.write(blankStr + "lambda——>" + lambdaMin + "-" + lambdaMax + "\n");
        fw.write(blankStr + "eta——>" + etaMin + "-" + etaMax + "\n");
        fw.write(blankStr + "minGap——>" + CommonRec_PSO.minGap + "\n");
        fw.write(starStr + "\n");
        fw.flush();
        
        ANLF_PSO_cc trainANLF = new ANLF_PSO_cc();
        trainANLF.last_Error = trainANLF.train(metrics, fw);
        
        System.out.println("r1::r2——>" + r1 + "::" + r2);
        System.out.println("Min training Error:\t\t\t" + trainANLF.last_Error);
        System.out.println("Min total training epochs:\t\t" + trainANLF.min_Round);
        System.out.println("Total Round:\t\t" + trainANLF.total_Round);
        System.out.println("Min total training time:\t\t" + trainANLF.minTotalTime);
        System.out.println("Min average training time:\t\t" + trainANLF.minTotalTime / trainANLF.min_Round);
        System.out.println("Total training time:\t\t" + trainANLF.total_Time);
        System.out.println("Average training time:\t\t" + trainANLF.total_Time / trainANLF.total_Round);
        
        fw.write("r1::r2——>" + r1 + "::" + r2 + "\n");
        fw.write("Min training Error:\t\t\t" + trainANLF.last_Error + "\n");
        fw.write("Min total training epochs:\t\t" + trainANLF.min_Round + "\n");
        fw.write("Total Round:\t\t" + trainANLF.total_Round + "\n");
        fw.write("Min total training time:\t\t" + trainANLF.minTotalTime + "\n");
        fw.write("Min average training time:\t\t" + trainANLF.minTotalTime / trainANLF.min_Round + "\n");
        fw.write("Total training time:\t\t" + trainANLF.total_Time + "\n");
        fw.write("Average training time:\t\t" + trainANLF.total_Time / trainANLF.total_Round + "\n");
        fw.flush();
        fw.close();
    }

    public double train(int metrics, FileWriter fw) throws IOException {

    	double startTime = System.currentTimeMillis();
    	if(metrics == CommonRec_PSO.RMSE)
    		FitnessRMSEcbest = 100;
    	else
    		FitnessMAEcbest = 100;
    	
    	// 初始化：将所有的rating估计值缓存起来，提高计算效率
        for (RTuple trainR : trainDataSet) {
            double ratingHat = dotMultiply(X[trainR.userID], Y[trainR.itemID]);
            trainR.ratingHat = ratingHat;
        } 
        double rho, tau;
        
        for(int q = 0; q < swarmNum; q++){
    		//step 01
    		// 每一轮开始时，将辅助矩阵元素置为0
            double lambda = particles[q][0];
            double eta = particles[q][0];
            for (int dim = 0; dim < featureDimension; dim++) {
            	// 将相关的辅助变量置为0
                resetAuxArray();

                for (RTuple trainR : trainDataSet) {
                    // 对于X，记录学习增量
                    // 分别计算对于用户的上下更新
                    X_U[trainR.userID] -= Y[trainR.itemID][dim] * (trainR.ratingHat - trainR.rating);
                    X_U[trainR.userID] += X[trainR.userID][dim] * Y[trainR.itemID][dim]
                            * Y[trainR.itemID][dim];
                    X_D[trainR.userID] += Y[trainR.itemID][dim] * Y[trainR.itemID][dim];
                    // 对于Y，记录学习增量
                    // 分别计算对于商品的上下更新
                    Y_U[trainR.itemID] -= X[trainR.userID][dim] * (trainR.ratingHat - trainR.rating);
                    Y_U[trainR.itemID] += Y[trainR.itemID][dim] * X[trainR.userID][dim]
                            * X[trainR.userID][dim];
                    Y_D[trainR.itemID] += X[trainR.userID][dim] * X[trainR.userID][dim];
                }

                for (int userID = 1; userID <= userMaxID; userID++){

                    X_C[userID] = X[userID][dim];

                    rho = lambda * userRSetSize[userID];
                    X_U[userID] += rho * P[userID][dim];
                    X_U[userID] -= Gamma_X[userID][dim];
                    if(rho != 0)
                        X_D[userID] += rho;
                    else
                        X_D[userID] += 1e-8;
                    X[userID][dim] = X_U[userID] / X_D[userID];

                    // 更新P矩阵
                    double tempP;
                    if(rho != 0)
                        tempP = X[userID][dim] + Gamma_X[userID][dim] / (rho);
                    else
                        tempP = X[userID][dim] + Gamma_X[userID][dim] / (1e-8);
                    if (tempP > 0)
                        this.P[userID][dim] = tempP;
                    else
                        this.P[userID][dim] = 0;
                    // 更新Gamma_X矩阵
                    Gamma_X[userID][dim] += eta * rho * (X[userID][dim] - P[userID][dim]);
                }

                for (int itemID = 1; itemID <= itemMaxID; itemID++) {

                    Y_C[itemID] = Y[itemID][dim];

                    tau = lambda * itemRSetSize[itemID];
                    Y_U[itemID] += tau * Q[itemID][dim];
                    Y_U[itemID] -= Gamma_Y[itemID][dim];
                    if(tau != 0)
                        Y_D[itemID] += tau;
                    else
                        Y_D[itemID] += 1e-8;
                    Y[itemID][dim] = Y_U[itemID] / Y_D[itemID];

                    // 更新Q矩阵
                    double tempQ;
                    if(tau != 0)
                        tempQ = Y[itemID][dim] + Gamma_Y[itemID][dim] / (tau);
                    else
                        tempQ = Y[itemID][dim] + Gamma_Y[itemID][dim] / (1e-8);
                    if (tempQ > 0)
                        Q[itemID][dim] = tempQ;
                    else
                        Q[itemID][dim] = 0;
                    // 更新Gamma_Y矩阵
                    Gamma_Y[itemID][dim] += eta * tau * (Y[itemID][dim] - Q[itemID][dim]);
                }

                // 根据X,Y矩阵的更新，对所有的rating估计值进行更新
                for (RTuple trainR : trainDataSet) {
                    double ratingHatNew = X[trainR.userID][dim] * Y[trainR.itemID][dim]
                            - X_C[trainR.userID] * Y_C[trainR.itemID];
                    trainR.ratingHat = trainR.ratingHat + ratingHatNew;
                }
            }
           
            if (metrics == CommonRec_PSO.RMSE) {
            	//Step 02 
            	FitnessRMSE[q] = valRMSE();
            	
            	//Step 03
                if(FitnessRMSE[q] < FitnessRMSEcbest)
                	FitnessRMSEcbest = FitnessRMSE[q];
                
                //step 04
                if(q == 0){
        			new_Fitness_U[0] = 1; //即不考虑初始化时FitnessRMSE[0]-last_FitnessRMSE[q-1]这种情况，从第2个粒子才开始算在本轮的贡献度，所以这里直接赋值任意正数
            	}else{
            		new_Fitness_U[q] = FitnessRMSE[q] - FitnessRMSE[q - 1];
            	}
        		new_Fitness_D[q] = -1; //直接赋值任意负数
        		
        		//step 05控制更新的方向
        		updateBestnFitness(new_Fitness_U[q], new_Fitness_D[q], q);
        		
        		//step 06控制什么时候结束
                updateBestRMSE(FitnessRMSE[q], q);  
            } else {
            	//Step 02 
            	FitnessMAE[q] = valMAE();
            	
            	//Step 03
                if(FitnessMAE[q] < FitnessMAEcbest)
                	FitnessMAEcbest = FitnessMAE[q];
                
                //step 04
                if(q == 0){
        			new_Fitness_U[0] = 1; //即不考虑初始化时FitnessRMSE[0]-last_FitnessRMSE[q-1]这种情况，从第2个粒子才开始算在本轮的贡献度，所以这里直接赋值任意正数
            	}else{
            		new_Fitness_U[q] = FitnessMAE[q] - FitnessMAE[q - 1];
            	}
        		new_Fitness_D[q] = -1; //直接赋值任意负数
        		
        		//step 05控制更新的方向
        		updateBestnFitness(new_Fitness_U[q], new_Fitness_D[q], q);
        		
        		//step 06控制什么时候结束
                updateBestMAE(FitnessMAE[q], q);  
            }  
    	}
		
    	if(metrics == CommonRec_PSO.RMSE){
    		last_FitnessRMSEcbest = FitnessRMSEcbest; 
    	}else{
    		last_FitnessMAEcbest = FitnessMAEcbest; 
    	}
    	min_Error = 100;
    	Random random = new Random(System.currentTimeMillis()); //所有轮共享1对
        r1 = random.nextDouble();
        r2 = random.nextDouble();
//    	r1 = 0.627053322360496;
//    	r2 = 0.95998583415187;
        for (int round = 1; round <= maxRound; round++){
//        	recordParticles();
//        	recordV();
        	
        	if(metrics == CommonRec_PSO.RMSE){
        		train_last_valRMSE = FitnessRMSE[swarmNum - 1];
        		FitnessRMSEcbest = 100;
        	}else{
        		train_last_valMAE = FitnessMAE[swarmNum - 1];
        		FitnessMAEcbest = 100;
        	}
        	
        	for (int q = 0; q < swarmNum; q++) {
        		//step 01
            	update_Particles_V(q);
            	
            	//step 02
            	double lambda = particles[q][0];
                double eta = particles[q][0];
            	for (int dim = 0; dim < featureDimension; dim++) {
                	// 将相关的辅助变量置为0
                    resetAuxArray();

                    for (RTuple trainR : trainDataSet) {
                        // 对于X，记录学习增量
                        // 分别计算对于用户的上下更新
                        X_U[trainR.userID] -= Y[trainR.itemID][dim] * (trainR.ratingHat - trainR.rating);
                        X_U[trainR.userID] += X[trainR.userID][dim] * Y[trainR.itemID][dim]
                                * Y[trainR.itemID][dim];
                        X_D[trainR.userID] += Y[trainR.itemID][dim] * Y[trainR.itemID][dim];
                        // 对于Y，记录学习增量
                        // 分别计算对于商品的上下更新
                        Y_U[trainR.itemID] -= X[trainR.userID][dim] * (trainR.ratingHat - trainR.rating);
                        Y_U[trainR.itemID] += Y[trainR.itemID][dim] * X[trainR.userID][dim]
                                * X[trainR.userID][dim];
                        Y_D[trainR.itemID] += X[trainR.userID][dim] * X[trainR.userID][dim];
                    }

                    for (int userID = 1; userID <= userMaxID; userID++){

                        X_C[userID] = X[userID][dim];

                        rho = lambda * userRSetSize[userID];
                        X_U[userID] += rho * P[userID][dim];
                        X_U[userID] -= Gamma_X[userID][dim];
                        if(rho != 0)
                            X_D[userID] += rho;
                        else
                            X_D[userID] += 1e-8;
                        X[userID][dim] = X_U[userID] / X_D[userID];

                        // 更新P矩阵
                        double tempP;
                        if(rho != 0)
                            tempP = X[userID][dim] + Gamma_X[userID][dim] / (rho);
                        else
                            tempP = X[userID][dim] + Gamma_X[userID][dim] / (1e-8);
                        if (tempP > 0)
                            this.P[userID][dim] = tempP;
                        else
                            this.P[userID][dim] = 0;
                        // 更新Gamma_X矩阵
                        Gamma_X[userID][dim] += eta * rho * (X[userID][dim] - P[userID][dim]);
                    }

                    for (int itemID = 1; itemID <= itemMaxID; itemID++) {

                        Y_C[itemID] = Y[itemID][dim];

                        tau = lambda * itemRSetSize[itemID];
                        Y_U[itemID] += tau * Q[itemID][dim];
                        Y_U[itemID] -= Gamma_Y[itemID][dim];
                        if(tau != 0)
                            Y_D[itemID] += tau;
                        else
                            Y_D[itemID] += 1e-8;
                        Y[itemID][dim] = Y_U[itemID] / Y_D[itemID];

                        // 更新Q矩阵
                        double tempQ;
                        if(tau != 0)
                            tempQ = Y[itemID][dim] + Gamma_Y[itemID][dim] / (tau);
                        else
                            tempQ = Y[itemID][dim] + Gamma_Y[itemID][dim] / (1e-8);
                        if (tempQ > 0)
                            Q[itemID][dim] = tempQ;
                        else
                            Q[itemID][dim] = 0;
                        // 更新Gamma_Y矩阵
                        Gamma_Y[itemID][dim] += eta * tau * (Y[itemID][dim] - Q[itemID][dim]);
                    }

                    // 根据X,Y矩阵的更新，对所有的rating估计值进行更新
                    for (RTuple trainR : trainDataSet) {
                        double ratingHatNew = X[trainR.userID][dim] * Y[trainR.itemID][dim]
                                - X_C[trainR.userID] * Y_C[trainR.itemID];
                        trainR.ratingHat = trainR.ratingHat + ratingHatNew;
                    }
                }
            	
            	
                if (metrics == CommonRec_PSO.RMSE) {
                	//step 03
                    // 计算本轮训练结束后，在验证集上的误差
                	FitnessRMSE[q] = valRMSE();
                	
                	//Step 04
                    if(FitnessRMSE[q] < FitnessRMSEcbest)
                		FitnessRMSEcbest = FitnessRMSE[q];
                    
                    //step 05控制什么时候结束
                    updateBestRMSE(FitnessRMSE[q], q);
                } else {
                	//step 03
                    // 计算本轮训练结束后，在验证集上的误差
                	FitnessMAE[q] = valMAE();
                	
                	//Step 04
                    if(FitnessMAE[q] < FitnessMAEcbest)
                		FitnessMAEcbest = FitnessMAE[q];
                    
                    //step 05控制什么时候结束
                    updateBestMAE(FitnessMAE[q], q);
                }
        	}
        	
        	for (int q = 0; q < swarmNum; q++) {
        		//step 01
        		if (metrics == CommonRec_PSO.RMSE) {
            		new_Fitness_U[q] = computingFitness_cc(FitnessRMSE,train_last_valRMSE, q, "U");
            		new_Fitness_D[q] = computingFitness_cc(FitnessRMSE,train_last_valRMSE, q, "D");
        		}else{ 			
            		new_Fitness_U[q] = computingFitness_cc(FitnessMAE,train_last_valMAE, q, "U");
            		new_Fitness_D[q] = computingFitness_cc(FitnessMAE,train_last_valMAE, q, "D");
        		}
            	
        		//step 02控制更新的方向
        		updateBestnFitness(new_Fitness_U[q], new_Fitness_D[q], q);
            }
            
        	if (metrics == CommonRec_PSO.RMSE) {
        		last_FitnessRMSEcbest = FitnessRMSEcbest;
        	}else{
        		last_FitnessMAEcbest = FitnessMAEcbest;
        	}
            
            double endTime = System.currentTimeMillis();
            cacheMinTotalTime += endTime - startTime;
            total_Time += endTime - startTime;
            
            if (metrics == CommonRec_PSO.RMSE) {
            	
                if (Math.abs(FitnessRMSEgbest - min_Error) < minGap) 
                    break;
                
                if (min_Error > FitnessRMSEgbest) { 
                    min_Error = FitnessRMSEgbest;
                    min_Round = round;
                    minTotalTime += cacheMinTotalTime;
                    cacheMinTotalTime = 0;
                }
                else if ((round - min_Round) >= delayCount) {
                    break;
                }
        	}else{
        		if (Math.abs(FitnessMAEgbest - min_Error) < minGap) 
                    break;
        		
        		if (min_Error > FitnessMAEgbest) { 
                    min_Error = FitnessMAEgbest;
                    min_Round = round;
                    minTotalTime += cacheMinTotalTime;
                    cacheMinTotalTime = 0;
                }
                else if ((round - min_Round) >= delayCount) {
                    break;
                }
        	}
            fw.write(min_Error + "\n");
            fw.flush();
            System.out.println(min_Error);
            total_Round += 1;
        }
        double lastErr;
        if (metrics == CommonRec_PSO.RMSE) {
            lastErr = testRMSE();
        } else {
            lastErr = testMAE();
        }
//        printResults_v2(metrics, featureDimension);
//      printParticles();
//      printV();
        
        return lastErr;  
    }
}
