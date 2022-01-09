package qfirst.model

import qasrl.data.QuestionLabel
import qasrl.data.VerbEntry

import cats.data.NonEmptyList
import cats.implicits._

import jjm.ling.en.InflectedForms
import jjm.metrics._
import jjm.metrics.HasMetrics.ops._

package object eval {

  def filterGold(minNumAnswers: Int, maxNumInvalid: Int) = (verb: VerbEntry) => {
    val (invalids, valids) = verb.questionLabels.toList.flatMap {
      case (questionString, qLabel) =>
        val judgments = qLabel.answerJudgments.toList.map(_.judgment)
        val numInvalid = judgments.filter(_.isInvalid).size
        val numAnswers = judgments.size
        if(numAnswers >= minNumAnswers) {
          if(numInvalid <= maxNumInvalid) Some(Right(questionString -> qLabel))
          else Some(Left(questionString -> qLabel))
        } else None
    }.separate
    invalids.toMap -> valids.toMap
  }

  val filterGoldNonDense = filterGold(3, 0)
  val filterGoldDense = filterGold(6, 1)

  def questionLabelIsValid(minNumAnswers: Int, maxNumInvalid: Int) = (qLabel: QuestionLabel) => {
    val judgments = qLabel.answerJudgments.toList.map(_.judgment)
    val numInvalid = judgments.filter(_.isInvalid).size
    val numAnswers = judgments.size
    if(numAnswers >= minNumAnswers) {
      if(numInvalid <= maxNumInvalid) true
      else false
    } else false
  }

  val questionLabelIsValidNonDense = questionLabelIsValid(3, 0)
  val questionLabelIsValidDense = questionLabelIsValid(6, 1)

  import cats.implicits._

  val sortSpec = {
    import Metric._
    import MapTree.SortQuery._
    val double = (mv: Metric) => mv match {
      case MetricMetadata(s) => 0.0
      case MetricBool(x) => if(x) 1.0 else 0.0
      case MetricInt(x) => x.toDouble
      case MetricDouble(x) => x
      case MetricIntOfTotal(x, _) => x.toDouble
    }
    val inc = value[String](double)
    val dec = value[String](double andThen (_ * -1))
    List(
      "predictions" :: "f1" :: inc,
      "full question" :: "f1" :: inc,
      "full question" :: "acc-lb" :: inc,
      "num predicted" :: inc
    )
  }
  def getMetricsString[M: HasMetrics](m: M) =
    m.getMetrics.toStringPrettySorted(identity, x => x.render, sortSpec)

  def evalBucketBounds(unsortedBounds: NonEmptyList[Int])(value: Int) = {
    val bounds = unsortedBounds.sorted
    if(value <= bounds.head) s"<=${bounds.head}"
    else bounds.toList.sliding(2).find(g => value > g(0) && value <= g(1)) match {
      case Some(g) => s"${g(0) + 1}-${g(1)}"
      case None => s">${bounds.last}"
    }
  }

  def verbFreq(getFreq: InflectedForms => Int, bounds: NonEmptyList[Int]) = (verb: VerbEntry) => {
    val freq = getFreq(verb.verbInflectedForms)
    evalBucketBounds(bounds)(freq)
  }

  def verbBucketers(getFreq: (InflectedForms => Int)) = Map(
    "verb-freq" -> verbFreq(
      getFreq,
      NonEmptyList.of(0, 10, 50, 150, 250, 500, 750, 1000))
  )

  def filterOrigAnnotationRound(verb: VerbEntry): VerbEntry = {
    val newQuestionLabels = scala.collection.immutable.SortedMap(
      verb.questionLabels.values.toList
        .filter(_.questionSources.exists(_.startsWith("turk-qasrl2.0-")))
        .map { qLabel =>
          val ajs = qLabel.answerJudgments.filter(aj =>
            !(aj.sourceId.endsWith("-expansion")) && !(aj.sourceId.endsWith("-eval"))
          )
          qLabel.copy(answerJudgments = ajs)
        }
        .filter(_.answerJudgments.nonEmpty)
        .map(qLabel => qLabel.questionString -> qLabel): _*
    )
    verb.copy(questionLabels = newQuestionLabels)
  }
}
