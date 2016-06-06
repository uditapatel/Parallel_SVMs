/**
 * Created by pateu14 (Patel Udita) on 4/17/2016.
 */

import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.myorg.Cascade3;
import java.io.IOException;
import java.util.Random;

public class Bagging1  {
    private static final Logger LOG = Logger.getLogger(Driver.class);
    public static void main(String[] args) throws Exception {
      Configuration firstConf = new Configuration();
      String[] otherArgs = new GenericOptionsParser(firstConf, args).getRemainingArgs();
      // otherArgs[0]: input path otherArgs[1]: output path otherArgs[2]: num of subset power of 2
      if (otherArgs.length < 2) {
          System.err.println("Usage: cascade-svm <in> <out> <subsets>");// how to call
          System.exit(2);
      }
      final double subsets = Double.valueOf(otherArgs[2]);
      firstConf.setInt("SUBSET_COUNT",(int)subsets);	// set a Int global value
      // pre-partition job count is two for two map-reds that partitions the data
      final int prepartitionJobCount = 2;
      final int cascadeJobCount = (int)(Math.log(subsets)/Math.log(2));
      Configuration[] prepartitionConfs = new Configuration[prepartitionJobCount];
      prepartitionConfs[0] = firstConf;
      prepartitionConfs[1] = new Configuration();
      Precascade1 pre1=new Precascade1();
      int res1 = ToolRunner.run(prepartitionConfs[0],pre1, otherArgs); //## change the class.
      System.out.println("The mapper exited with : "+res1);
      prepartitionConfs[1].setInt("SUBSET_COUNT",(int)subsets);
      prepartitionConfs[1].setInt("TOTAL_RECORD_COUNT",
              (int)pre1.getJob().getCounters().findCounter("trainingDataStats","TOTAL_RECORD_COUNT").getValue());
      Precascade2 pre2=new Precascade2();
      int res2= ToolRunner.run(prepartitionConfs[1],pre2,otherArgs);
      Configuration baggingJobConf = new Configuration();
      DistributedCache.addFileToClassPath( new Path("/user/cloudera/cascade/libsvm3.2.jar"), baggingJobConf);
      DistributedCache.addFileToClassPath( new Path("/user/cloudera/cascade/javaml-0.1.7.jar"), baggingJobConf);
      baggingJobConf.set("USER_OUTPUT_PATH", otherArgs[1]);
      System.out.println("===== Beginning job for bagging :=====");
      ToolRunner.run(baggingJobConf,new Bagging(),args);
    }
  }
