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


public class Test {
    public static void main(String[] args ) {
   SparkSession spark = SparkSession
   .builder()
   .appName("Data")
   .getOrCreate();

   Dataset<Row> df = spark.read().option("header","true").csv("penguins_size.csv");
   
   StringIndexer indexer = new StringIndexer()

   .setInputCols(new String[] {"species", "island", "sex","culmen_length_mm","culmen_depth_mm","flipper_length_mm","body_mass_g"})
   .setOutputCols(new String[] {"speciesIndex", "islandIndex", "sexIndex","culmen_length_mmIndex","culmen_depth_mmIndex","flipper_length_mmIndex","body_mass_gIndex"});
   Dataset<Row> indexed = indexer.fit(df).transform(df);
   //indexed.setHandleInvalid("keep");
   //indexed.show();
   //indexed.printSchema();
   //indexed.describe().show();

   VectorAssembler assembler = new VectorAssembler()
  .setInputCols(new String[] {"culmen_length_mmIndex", "culmen_depth_mmIndex","flipper_length_mmIndex", "body_mass_gIndex"})
  .setOutputCol("features");
Dataset<Row> assembled = assembler.transform(indexed);
//assembled.select("features").show();

StandardScaler scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(false);
StandardScalerModel scalerModel = scaler.fit(assembled);
Dataset<Row> scaled = scalerModel.transform(assembled);
//scaled.select("scaledFeatures").show();
//scaled.describe().show();

VectorAssembler assembler_fin = new VectorAssembler()
  .setInputCols(new String[] {"scaledFeatures", "islandIndex", "sexIndex"})
  .setOutputCol("big_features");
Dataset<Row> data = assembler_fin.transform(scaled).select("speciesIndex", "big_features");
data = data.withColumnRenamed("speciesIndex", "label");
data = data.withColumnRenamed("big_features", "features");
// data.show();
   
LogisticRegression lr = new LogisticRegression();
LogisticRegressionModel lrModel = lr.fit(data);

JavaRDD<Row> data_rdd = data.toJavaRDD();
JavaPairRDD< Object, Object> predictionAndLabels = data_rdd.mapToPair(p ->
  new Tuple2<>(lrModel.predict(p.getAs(1)), p.getAs(0)));
MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());



    }
}
