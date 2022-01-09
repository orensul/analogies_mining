package qfirst.clause.ext.demo
import qfirst.clause.ext._

import qasrl.crowd.QASRLValidationWorkerInfoSummary

import jjm.DotKleisli

import io.circe.{Encoder, Decoder}
import io.circe.generic.JsonCodec

@JsonCodec case class ClausalPrompt[SID](sentenceId: SID)

@JsonCodec case class ClausalAjaxRequest[SID](workerIdOpt: Option[String], id: SID) {
  type Out = ClausalAjaxResponse
}
object ClausalAjaxRequest {
  implicit def clausalAjaxRequestDotEncoder[SID] = new DotKleisli[Encoder, ClausalAjaxRequest[SID]] {
    def apply(req: ClausalAjaxRequest[SID]) = implicitly[Encoder[ClausalAjaxResponse]]
  }
  implicit def clausalAjaxRequestDotDecoder[SID] = new DotKleisli[Decoder, ClausalAjaxRequest[SID]] {
    def apply(req: ClausalAjaxRequest[SID]) = implicitly[Decoder[ClausalAjaxResponse]]
  }
}

@JsonCodec case class ClausalAjaxResponse(
  workerInfoOpt: Option[QASRLValidationWorkerInfoSummary],
  predictions: ClausalSentencePrediction
)
