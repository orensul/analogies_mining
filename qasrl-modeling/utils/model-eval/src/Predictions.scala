package qfirst.model.eval

import jjm.ling.en.InflectedForms

import io.circe.generic.JsonCodec

@JsonCodec case class VerbPrediction[A](
  verbIndex: Int,
  verbInflectedForms: InflectedForms,
  beam: A
)

@JsonCodec case class SentencePrediction[A](
  sentenceId: String,
  sentenceTokens: Vector[String],
  verbs: List[VerbPrediction[A]]
)
