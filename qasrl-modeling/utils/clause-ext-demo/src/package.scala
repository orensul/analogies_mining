package qfirst.clause.ext

import qfirst.model.eval.SentencePrediction
import qfirst.model.eval.protocols._

import qasrl.crowd.QASRLEvaluationSettings

package object demo {

  type ClausalSentencePrediction = SentencePrediction[
    FactoringProtocol.Beam[QfirstBeamItem[Map[String, String]]]
  ]

  def dollarsToCents(d: Double): Int = math.round(100 * d).toInt

  implicit val settings = new QASRLEvaluationSettings {}
}
