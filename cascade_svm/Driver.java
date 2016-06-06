
/**
 * Created by pateu14 (patel udita)on 4/10/2016.
 */

import java.io.IOException;
import java.util.Random;
import java.util.Vector;
import java.util.StringTokenizer;
import org.apache.hadoop.filecache.DistributedCache;
import libsvm.*;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.log4j.Logger;

public class Cascade2  {
    private static final Logger LOG = Logger.getLogger(Cascade2.class);

    public static void main(String[] args) throws Exception {

        Configuration firstConf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(firstConf, args).getRemainingArgs();
        // otherArgs[0]: input path otherArgs[1]: output path otherArgs[2]: num of subset power of 2
        if (otherArgs.length < 2) {
            System.err.println("Usage: cascade-svm <in> <out> <subsets>");
            System.exit(2);
        }

        final double subsets = Double.valueOf(otherArgs[2]);

        if (subsets % 2 != 0){
            System.err.println("The number of subsets for a binary cascade svm should be a power of 2.");
            System.exit(2);
        }

        firstConf.setInt("SUBSET_COUNT",(int)subsets);	// set a Int global value
        // pre-partition job count is two for two map-reds that partitions the data
        final int prepartitionJobCount = 2;
        final int cascadeJobCount = (int)(Math.log(subsets)/Math.log(2));

        Configuration[] prepartitionConfs = new Configuration[prepartitionJobCount];
        prepartitionConfs[0] = firstConf;
        prepartitionConfs[1] = new Configuration();
        Precascade1 pre1=new Precascade1();
        int res1 = ToolRunner.run(prepartitionConfs[0],pre1, otherArgs);
        System.out.println("The mapper exited with : "+res1);

        prepartitionConfs[1].setInt("SUBSET_COUNT",(int)subsets);
        prepartitionConfs[1].setInt("TOTAL_RECORD_COUNT",
                (int)pre1.getJob().getCounters().findCounter("trainingDataStats","TOTAL_RECORD_COUNT").getValue());
        Precascade2 pre2=new Precascade2();
        int res2= ToolRunner.run(prepartitionConfs[1],pre2,otherArgs);

        /*** cascade job starts ***/

        Configuration[] cascadeConfs = new Configuration[cascadeJobCount];

	      for(int i=0;i<cascadeJobCount;i++){
            cascadeConfs[i]=new Configuration();
        }

        cascadeConfs[cascadeJobCount-1].set("USER_OUTPUT_PATH", otherArgs[1]);
        for(int remainingConfs = 0; remainingConfs < cascadeJobCount; remainingConfs++) {
            cascadeConfs[remainingConfs] = new Configuration();
            cascadeConfs[remainingConfs].setInt("SUBSET_COUNT",(int)subsets);
	          DistributedCache.addFileToClassPath( new Path("/user/cloudera/cascade/libsvm3.2.jar"), cascadeConfs[remainingConfs]);
	          DistributedCache.addFileToClassPath( new Path("/user/cloudera/cascade/javaml-0.1.7.jar"), cascadeConfs[remainingConfs]);
        }
        String[] cargs=new String[otherArgs.length];
        for(int i=0;i<otherArgs.length-1;i++){
            cargs[i]=otherArgs[i];
        }
        for(int jobItr = 0; jobItr < cascadeJobCount; jobItr++){
            System.out.println("===== Beginning job for layer " + (jobItr+1) + " =====");
            cargs[2]=String.valueOf((jobItr+1));
	          System.out.println("===== Beginning job for layer " + cargs[2] + " =====");
            if(jobItr != cascadeJobCount-1){
                ToolRunner.run(cascadeConfs[jobItr],new Midcascade(),cargs);
            } else {
                ToolRunner.run(cascadeConfs[jobItr],new Lastcascade(),cargs);
            }
        }
        System.exit(res2);
    }
}
