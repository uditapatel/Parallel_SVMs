class Bagging extends Configured implements  Tool{
    public int run(String[] args) throws Exception {
      Job baggingJob = Job.getInstance(this.getConf(),"Bagging SVM: Base models training");
      baggingJob.setJarByClass(this.getClass());
      baggingJob.setNumReduceTasks(0);
      baggingJob.setMapperClass(BaseModelOutputMapper.class);
      baggingJob.setOutputKeyClass(NullWritable.class);
      baggingJob.setOutputValueClass(Text.class);
      baggingJob.setInputFormatClass(TrainingSubsetInputFormat.class);
      FileInputFormat.addInputPath(baggingJob, new Path(args[1]+"/base-model-subsets"));
      FileOutputFormat.setOutputPath(baggingJob, new Path(args[1]+"/base-model-SVs"));
      return baggingJob.waitForCompletion(true) ? 0 : 1;
    }
    public static class BaseModelOutputMapper extends Mapper<NullWritable, Text, NullWritable, Text>{
      private static final String svm_type_table[] = {
        "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",
      };

      static final String kernel_type_table[]= {
        "linear","polynomial","rbf","sigmoid","precomputed"
      };

      // An identical implementation of svm.svm_save_model in LIBSVM,
      // different in that the file is saved to HDFS instead of a local path.
      private void saveModelToHdfs(svm_model model, String pathStr, int taskId, Context context){
        try {
          FileSystem fs = FileSystem.get(context.getConfiguration());
          Path file = new Path(fs.getHomeDirectory(),"/user/cloudera/bagging1/output/base-models/model-"+taskId+".model");
          FSDataOutputStream fos = fs.create(file);
          svm_parameter param = model.param;
          fos.writeBytes("svm_type "+svm_type_table[param.svm_type]+"\n");
          fos.writeBytes("kernel_type "+kernel_type_table[param.kernel_type]+"\n");

          if(param.kernel_type == svm_parameter.POLY)
              fos.writeBytes("degree "+param.degree+"\n");
          if(param.kernel_type == svm_parameter.POLY ||
                  param.kernel_type == svm_parameter.RBF ||
                  param.kernel_type == svm_parameter.SIGMOID)
              fos.writeBytes("gamma "+param.gamma+"\n");

          if(param.kernel_type == svm_parameter.POLY ||
                  param.kernel_type == svm_parameter.SIGMOID)
              fos.writeBytes("coef0 "+param.coef0+"\n");

          int nr_class = model.nr_class;
          int l = model.l;
          fos.writeBytes("nr_class "+nr_class+"\n");
          fos.writeBytes("total_sv "+l+"\n");

          fos.writeBytes("rho");
          for(int i=0;i<nr_class*(nr_class-1)/2;i++)
              fos.writeBytes(" "+model.rho[i]);
          fos.writeBytes("\n");

          if(model.label != null) {
              fos.writeBytes("label");
              for(int i=0;i<nr_class;i++)
                  fos.writeBytes(" "+model.label[i]);
              fos.writeBytes("\n");
          }

          if(model.probA != null) { // regression has probA only
              fos.writeBytes("probA");
              for(int i=0;i<nr_class*(nr_class-1)/2;i++)
                  fos.writeBytes(" "+model.probA[i]);
              fos.writeBytes("\n");
          }

          if(model.probB != null) {
              fos.writeBytes("probB");
              for(int i=0;i<nr_class*(nr_class-1)/2;i++)
                  fos.writeBytes(" "+model.probB[i]);
              fos.writeBytes("\n");
          }

          if(model.nSV != null) {
              fos.writeBytes("nr_sv");
              for(int i=0;i<nr_class;i++)
                  fos.writeBytes(" "+model.nSV[i]);
              fos.writeBytes("\n");
          }

          fos.writeBytes("SV\n");
          double[][] sv_coef = model.sv_coef;
          svm_node[][] SV = model.SV;

          for(int i=0;i<l;i++) {
              for(int j=0;j<nr_class-1;j++)
                  fos.writeBytes(sv_coef[j][i]+" ");

              svm_node[] p = SV[i];
              if(param.kernel_type == svm_parameter.PRECOMPUTED)
                  fos.writeBytes("0:"+(int)(p[0].value));
              else
                  for(int j=0;j<p.length;j++)
                      fos.writeBytes(p[j].index+":"+p[j].value+" ");
              fos.writeBytes("\n");
          }
          fos.close();
          } catch (IOException ioe) {
              throw new RuntimeException(ioe);
          }
      }

      private Text supportVector = new Text();
  
      public void map(NullWritable offset, Text trainingSubset,Context context) throws IOException, InterruptedException {
        String[] svRecords = trainingSubset.toString().split("\n");
        Dataset ds=new DefaultDataset();
        /* Can be put in method : converts string[] to double []*/
          for(int i=0;i<svRecords.length;i++){
            String[] record = svRecords[i].split(",");
            int class_label=Integer.parseInt(record[0]);
            double[] features=new double[record.length-1];
            for(int j=1;j<record.length;j++){
                features[j-1]=Double.parseDouble(record[j]);
            }
            Instance instance = new DenseInstance(features,class_label);
            ds.add(instance);
            }
            // train with data
            LibSVM_modified javamlsmo =new LibSVM_modified();
            javamlsmo.buildClassifier(ds);
            String userOutputPathStr = context.getConfiguration().get("USER_OUTPUT_PATH");
            int taskId = context.getTaskAttemptID().getTaskID().getId();
            saveModelToHdfs(javamlsmo.get_model(),userOutputPathStr,taskId,context);
            int[] svIndices = javamlsmo.get_model().sv_indices;
            for(int i=0; i<svIndices.length; i++) {
                supportVector.set(svRecords[svIndices[i]-1]);
                context.write(NullWritable.get(), supportVector);
            }
        }
    }
}

