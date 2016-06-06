/**
 * Created by pateu14(Patel Udita) on 4/17/2016.
 */

import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Utils;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import java.io.*;
import java.net.URI;
import java.util.Map;
public class Driver  {
    private static final Logger LOG = Logger.getLogger(Iter1.class);
    public static void main(String[] args) throws Exception {
      Configuration firstConf = new Configuration();
      String[] otherArgs = new GenericOptionsParser(firstConf, args).getRemainingArgs();
      // otherArgs[0]: input path otherArgs[1]: output path otherArgs[2]: num of subset power of 2
      if (otherArgs.length < 2) {
          System.err.println("Usage: iterative-svm <in> <out> <subsets>");// how to call
          System.exit(2);
      }
      final double subsets = Double.valueOf(otherArgs[2]);
      firstConf.setInt("SUBSET_COUNT",(int)subsets);	// set a Int global value
      // pre-partition job count is two for two map-reds that partitions the data
      final int prepartitionJobCount = 2;
      // log(8) / log(2) =3
      // final int cascadeJobCount = (int)(Math.log(subsets)/Math.log(2));
      Configuration[] prepartitionConfs = new Configuration[prepartitionJobCount];
      prepartitionConfs[0] = firstConf;
      prepartitionConfs[1] = new Configuration();
      Preiterative1 pre1=new Preiterative1();
      int res1 = ToolRunner.run(prepartitionConfs[0],pre1, otherArgs); //## change the class.
      System.out.println("The mapper exited with : "+res1);
      prepartitionConfs[1].setInt("SUBSET_COUNT",(int)subsets);
      prepartitionConfs[1].setInt("TOTAL_RECORD_COUNT",
              (int)pre1.getJob().getCounters().findCounter("trainingDataStats","TOTAL_RECORD_COUNT").getValue());
      Preiterative2 pre2=new Preiterativee2();
      int res2= ToolRunner.run(prepartitionConfs[1],pre2,otherArgs);
      /*** Iterative job starts ***/
      System.out.println("===== Beginning job for iter number :=====");
      long olderrorsum=Long.MAX_VALUE;
      long newerrorsum=Long.MAX_VALUE;
      long temp;
      int i=1;
      String[] iargs=new String[4];
      for(int j=0;j<otherArgs.length;j++){
          iargs[j]=otherArgs[j];
      }
      do {
        iargs[3]=String.valueOf(i);
        Configuration IJobConf = new Configuration();
        DistributedCache.addCacheFile(new URI("/user/cloudera/iter1/gsvs/global_sv.csv"),IJobConf);
        DistributedCache.addFileToClassPath( new Path("/user/cloudera/cascade/libsvm3.2.jar"), IJobConf);
        DistributedCache.addFileToClassPath( new Path("/user/cloudera/cascade/javaml-0.1.7.jar"), IJobConf);
        IJobConf.set("USER_OUTPUT_PATH", iargs[1]);
        Itergsv ig = new Itergsv();
        ToolRunner.run(IJobConf,ig, iargs);
        olderrorsum=newerrorsum;
        newerrorsum=ig.ijob.getCounters().findCounter("TOTAL_MIS_CLF","errorsum").getValue();
        System.out.println("old value : "+olderrorsum);
        System.out.println("new value :"+newerrorsum);
        i++;
      }while(newerrorsum<olderrorsum&&i<3);
    }
  }
