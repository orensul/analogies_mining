package qfirst.clause.ext.demo
import qfirst.clause.ext._

import qasrl.crowd.QASRLEvaluationSettings

import spacro.tasks._

import io.circe.{Encoder, Decoder}

import org.scalajs.jquery.jQuery

import japgolly.scalajs.react.vdom.html_<^.VdomTag

abstract class ClausalDispatcher[SID : Encoder : Decoder](
  implicit settings: QASRLEvaluationSettings
) extends TaskDispatcher {

  def evaluationInstructions: VdomTag

  lazy val evalClient = new ClausalClient[SID](evaluationInstructions)

  final override lazy val taskMapping = Map[String, () => Unit](
    settings.evaluationTaskKey -> (() => evalClient.main)
  )

}
