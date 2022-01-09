package qfirst.clause.ext.demo
import qfirst.clause.ext._

import cats._
import cats.implicits._

import qasrl.crowd._
import qasrl.labeling._
import qasrl.bank.SentenceId

import spacro._
import spacro.tasks._

import jjm.LowerCaseString
import jjm.ling._
import jjm.implicits._
import jjm.corenlp.PosTagger

import akka.actor._
import akka.stream.scaladsl.Flow
import akka.stream.scaladsl.Source

import scala.concurrent.duration._
import scala.language.postfixOps

import scala.util.Try

import java.io.StringReader
import java.nio.file.{Files, Path, Paths}

import scala.util.Try
import scala.util.Random

class ClausalExample(
  val label: String = "trial",
  // qasrlBankPath: Path,
  clausePredictionPath: Path,
  frozenEvaluationHITTypeId: Option[String] = None
)(implicit config: TaskConfig) {

  val resourcePath = java.nio.file.Paths.get("datasets")

  import java.nio.file.{Files, Path, Paths}
  private[this] val liveDataPath = Paths.get(s"data/example/$label/live")
  implicit val liveAnnotationDataService = new FileSystemAnnotationDataService(liveDataPath)

  // val staticDataPath = Paths.get(s"data/example/$label/static")

  // def saveOutputFile(name: String, contents: String): Try[Unit] = Try {
  //   val path = staticDataPath.resolve("out").resolve(name)
  //   val directory = path.getParent
  //   if (!Files.exists(directory)) {
  //     Files.createDirectories(directory)
  //   }
  //   Files.write(path, contents.getBytes())
  // }

  // def loadOutputFile(name: String): Try[List[String]] = Try {
  //   val path = staticDataPath.resolve("out").resolve(name)
  //   import scala.collection.JavaConverters._
  //   Files.lines(path).iterator.asScala.toList
  // }

  // def loadInputFile(name: String): Try[List[String]] = Try {
  //   val path = staticDataPath.resolve("in").resolve(name)
  //   import scala.collection.JavaConverters._
  //   Files.lines(path).iterator.asScala.toList
  // }

  val sentencePredictions = {
    import ammonite.ops._
    import io.circe.jawn
    read.lines(ammonite.ops.Path(clausePredictionPath, pwd)).toList
      .traverse(jawn.decode[ClausalSentencePrediction])
      .map(_.map(pred => SentenceId.fromString(pred.sentenceId) -> pred).toMap)
  }.right.get

  def getSentenceTokens = (id: SentenceId) => {
    val toks = sentencePredictions(id).sentenceTokens
      .map(Token(_))
    PosTagger.posTag(Text.addIndices(toks))
  }

  val allPrompts = sentencePredictions.keys.toVector.map(ClausalPrompt(_))

  def numGenerationAssignmentsForPrompt(p: ClausalPrompt[SentenceId]) = 1

  lazy val experiment = new ClausalAnnotationPipeline(
    allPrompts,
    getSentenceTokens,
    sentencePredictions,
    numGenerationAssignmentsForPrompt,
    frozenEvaluationHITTypeId = frozenEvaluationHITTypeId,
    validationAgreementDisqualTypeLabel = None
  )

  // def saveAnnotationData[A](
  //   filename: String,
  //   ids: Vector[SentenceId],
  //   genInfos: List[HITInfo[QASRLGenerationPrompt[SentenceId], List[VerbQA]]],
  //   valInfos: List[HITInfo[QASRLValidationPrompt[SentenceId], List[QASRLValidationAnswer]]],
  //   labelMapper: QuestionLabelMapper[String, A],
  //   labelRenderer: A => String
  // ) = {
  //   saveOutputFile(
  //     s"$filename.tsv",
  //     DataIO.makeQAPairTSV(
  //       ids.toList,
  //       SentenceId.toString,
  //       genInfos,
  //       valInfos,
  //       labelMapper,
  //       labelRenderer
  //     )
  //   )
  // }

  // def saveAnnotationDataReadable(
  //   filename: String,
  //   ids: Vector[SentenceId],
  //   genInfos: List[HITInfo[QASRLGenerationPrompt[SentenceId], List[VerbQA]]],
  //   valInfos: List[HITInfo[QASRLValidationPrompt[SentenceId], List[QASRLValidationAnswer]]]
  // ) = {
  //   saveOutputFile(
  //     s"$filename.tsv",
  //     DataIO.makeReadableQAPairTSV(
  //       ids.toList,
  //       SentenceId.toString,
  //       identity,
  //       genInfos,
  //       valInfos,
  //       (id: SentenceId, qa: VerbQA, responses: List[QASRLValidationAnswer]) =>
  //         responses.forall(_.isAnswer)
  //     )
  //   )
  // }
}
