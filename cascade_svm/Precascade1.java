class Precascade1 extends Configured implements Tool{

    private Job job;

    public Job getJob(){return job;}
    public int run(String[] args) throws Exception {
        job = Job.getInstance(this.getConf(), "Cascade SVM: Partitioning training data, Phase 1");
        job.setJarByClass(this.getClass());
        job.setNumReduceTasks(0);
        // Use TextInputFormat, the default unless job.setInputFormatClass is used
        job.setMapperClass(PreStatCounterMapper.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        // The first mapper will read data from file and put it into tmp folder to be used by second mapper. No reducers Used.
        FileOutputFormat.setOutputPath(job, new Path(args[1]+"/tmp"));
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static class PreStatCounterMapper extends Mapper<Object, Text, NullWritable, Text>{

      public void map(Object offset, Text v_trainingData,Context context) throws IOException, InterruptedException {
        context.getCounter("trainingDataStats","TOTAL_RECORD_COUNT").increment(1);
        String dataStr = v_trainingData.toString();
        String label = dataStr.substring(0, dataStr.indexOf(","));
        // Can only recognize labels 0-9 for each digit
        context.getCounter("trainingDataStats","CLASS_"+label+"_COUNT").increment(1);
        context.write(NullWritable.get(), v_trainingData);
      }
    }

}
