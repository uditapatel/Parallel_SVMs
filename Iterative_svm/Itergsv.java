class Itergsv extends Configured implements  Tool {
    private Job ijob;
    public int run(String[] args) throws Exception {
      ijob = Job.getInstance(this.getConf(), "Iterative SVM: training");
      ijob.setJarByClass(this.getClass());
      ijob.setNumReduceTasks(Integer.parseInt(args[2]));
      ijob.setReducerClass(GSVReducer.class);
      ijob.setMapperClass(GSVOutputMapper.class);
      ijob.setMapOutputKeyClass(IntWritable.class);
      ijob.setMapOutputValueClass(Text.class);
      ijob.setOutputKeyClass(NullWritable.class);
      ijob.setOutputValueClass(Text.class);
      ijob.setInputFormatClass(TrainingSubsetInputFormat.class);
      FileInputFormat.addInputPaths(ijob, args[1] + "/subsets");
      FileOutputFormat.setOutputPath(ijob, new Path(args[1] + "/gsv/gsv-"+args[3]));
      return ijob.waitForCompletion(true) ? 0 : 1;
    }

    public static void moveFiles(Path from, Path to, Configuration conf) throws IOException {
      FileSystem fs = from.getFileSystem(conf); // get file system
      for (FileStatus status : fs.listStatus(from)) { // list all files in 'from' folder
        Path file = status.getPath(); // get path to file in 'from' folder
        Path dst = new Path(to, file.getName()); // create new file name
        fs.rename(file, dst); // move file from 'from' folder to 'to' folder
      }
    }


    public static class GSVOutputMapper extends Mapper<Object, Text, IntWritable, Text> {
      private Text mergedData = new Text();
      private IntWritable partitionIndex = new IntWritable();
      public void map(Object offset, Text wholeSubset,Context context) throws IOException, InterruptedException {
        String[] subsetRecords = wholeSubset.toString().split("\n");
        for (int i = 0; i < subsetRecords.length; i++) {
          mergedData.set(subsetRecords[i]);
          int taskId = context.getTaskAttemptID().getTaskID().getId();
          partitionIndex.set((int) Math.floor(taskId));
          context.write(partitionIndex, mergedData);
        }
      }
    }
    public static class GSVReducer extends Reducer<IntWritable, Text, NullWritable, Text> {
      private Text gSvs = new Text();
      private Path getPath=null;
      public void setup(Context context) throws IOException {
        Configuration conf = context.getConfiguration();
        FileSystem fs = FileSystem.get(conf);
        URI[] cacheFiles = DistributedCache.getCacheFiles(conf);
        Path getPath = new Path(cacheFiles[0].getPath());
      }
      public void reduce(IntWritable subsetId, Iterable<Text> v_subsetTrainingDataset,Context context) throws IOException, InterruptedException {
        Dataset ds = new DefaultDataset();
        for (Text trainingData : v_subsetTrainingDataset) {
            String[] record = trainingData.toString().split(",");
            int class_label = Integer.parseInt(record[0]);
            double[] features = new double[record.length - 1];
            for (int j = 1; j < record.length; j++) {
                features[j - 1] = Double.parseDouble(record[j]);
            }
            Instance instance = new DenseInstance(features, class_label);
            ds.add(instance);
        }
        FileSystem fs = FileSystem.get(context.getConfiguration());
        URI[] localFiles = DistributedCache.getCacheFiles(context.getConfiguration());
        Path pt=null;
        for(URI temp: localFiles){
            if(temp.getPath().contains("global_sv.csv")){
                pt=new Path(fs.getHomeDirectory(),temp.getPath());
            }
        }
        Dataset ds_gsv=new DefaultDataset();
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
        try {
            String line;
            //line=br.readLine();
            while ((line =br.readLine())!=null&&line.length()>1){
              System.out.println(line);
              String[] record = line.split(",");
              int class_label = Integer.parseInt(record[0]);
              double[] features = new double[record.length - 1];
              for (int j = 1; j < record.length; j++) {
                features[j - 1] = Double.parseDouble(record[j]);
              }
              Instance instance = new DenseInstance(features, class_label);
              ds_gsv.add(instance);
            }
            } finally {
                // you should close out the BufferedReader
                br.close();
            }
            ds.addAll(ds_gsv);
            LibSVM_modified javamlsmo = new LibSVM_modified();
            javamlsmo.buildClassifier(ds);

            Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(javamlsmo, ds);
            for (Object o : pm.keySet())
                context.getCounter("TOTAL_MIS_CLF", "errorsum").increment((long) (pm.get(o).getErrorRate()*100));
            int[] svIndices = javamlsmo.get_model().sv_indices;
            // BufferedWriter bw=new BufferedWriter(new OutputStreamWriter(fs.open(pt)));
            //Configuration conf = context.getConfiguration();
            FSDataOutputStream fos =fs.append(pt);
            for (int i = 0; i < svIndices.length; i++) {
                gSvs.set(ds.get(svIndices[i] - 1).toString());
                if (!ds_gsv.contains(ds.get(svIndices[i] - 1))){
                    context.write(NullWritable.get(), gSvs);
                    fos.writeBytes(ds.get(svIndices[i]-1).getClass().toString()+","+ds.get(svIndices[i]-1).values().toString().substring(2,ds.get(svIndices[i]-1).values().toString().length()-4));
                }
            }
            fos.close();
          }
        }
    }
