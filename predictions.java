import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import scala.Tuple2;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;


public class Test {
    public static void main(String[] args ) {
   SparkSession spark = SparkSession
   .builder()
   .appName("Data")
   .getOrCreate();

   Dataset<Row> df_raw = spark.read().option("header","true").csv("penguins_size.csv");
   
   StringIndexer indexer = new StringIndexer()
  .setInputCols(new String[] {"species", "island", "sex","culmen_length_mm","culmen_depth_mm","flipper_length_mm","body_mass_g"})
  .setOutputCols(new String[] {"label", "islandIndex", "sexIndex","culmen_length_mmIndex","culmen_depth_mmIndex","flipper_length_mmIndex","body_mass_gIndex"});

Dataset<Row> indexed = indexer.fit(df_raw).transform(df_raw);

VectorAssembler assembler = new VectorAssembler()
  .setInputCols(new String[] {"islandIndex", "culmen_length_mmIndex", "culmen_depth_mmIndex", "flipper_length_mmIndex", "body_mass_gIndex", "sexIndex"})
  .setOutputCol("features_pre");
Dataset<Row> data = assembler.transform(indexed).select("label", "features_pre");
data.show();

StandardScaler scaler = new StandardScaler()
      .setInputCol("features_pre")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(false);
LogisticRegression lr = new LogisticRegression();

Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {scaler, lr});

Dataset<Row>[] data_split = data.randomSplit(new double[] {0.8, 0.2}, 12345);
Dataset<Row> train = data_split[0];
Dataset<Row> test = data_split[1];

PipelineModel model = pipeline.fit(train);

Dataset<Row> predictions = model.transform(test);

JavaRDD<Row> predictions_rdd = predictions.toJavaRDD();
JavaPairRDD< Object, Object> predictionAndLabels = predictions_rdd.mapToPair(p ->
  new Tuple2<>(p.getAs(5), p.getAs(0)));
MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
   

    }
}
