package qfirst.frames.annotation

import qasrl.bank.SentenceId

import qfirst.frames.ArgumentSlot
import qfirst.frames.Frame

import io.circe.generic.JsonCodec

@JsonCodec case class ClauseChoice(
  frame: Frame,
  argumentSlot: ArgumentSlot)
object ClauseChoice {
  def make(p: (Frame, ArgumentSlot)) = ClauseChoice(p._1, p._2)
}

@JsonCodec case class ClauseAmbiguity(
  sentenceId: SentenceId,
  verbIndex: Int,
  questionString: String,
  structures: Set[ClauseChoice])

@JsonCodec case class ClauseResolution(
  ambiguity: ClauseAmbiguity,
  choiceOpt: Option[Set[ClauseChoice]]
)

trait ClauseAnnotationService[F[_]] { self =>
  def getResolution(isFull: Boolean, index: Int): F[ClauseResolution]
  def saveResolution(isFull: Boolean, index: Int, choice: Set[ClauseChoice]): F[ClauseResolution]
  // final def mapK[G[_]](f: F ~> G): ClauseAnnotationService[G] = new ClauseAnnotationService[G] {
  //   def getResolution(isFull: Boolean, index: Int): G[ClauseResolution] =
  //     f(self.getResolution(isFull, index))
  //   def saveResolution(isFull: Boolean, index: Int, choice: ClauseChoice): G[Option[ClauseChoice]] =
  //     f(self.saveResolution(isFull, index))
  // }
}
