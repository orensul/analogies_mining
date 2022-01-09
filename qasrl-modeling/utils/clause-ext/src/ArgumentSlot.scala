package qfirst.clause.ext

import jjm.{DependentEncoder, DependentDecoder}
import jjm.LowerCaseString
import jjm.implicits._

sealed trait ArgumentSlot { type Out }
case object Subj extends ArgumentSlot { type Out = Noun }
case object Obj extends ArgumentSlot { type Out = Noun }
case object Prep1 extends ArgumentSlot { type Out = Preposition }
case object Prep2 extends ArgumentSlot { type Out = Preposition }
case object Misc extends ArgumentSlot { type Out = NonPrepArgument }
case class Adv(wh: LowerCaseString) extends ArgumentSlot { type Out = Unit }

object ArgumentSlot {
  type Aux[A] = ArgumentSlot { type Out = A }

  def allAdvSlots =
    List("when", "where", "why", "how", "how long", "how much").map(s => Adv(s.lowerCase))

  def toString(slot: ArgumentSlot): String = slot match {
    case Subj    => "subj"
    case Obj     => "obj"
    case Prep1   => "prep1"
    case Prep2   => "prep2"
    case Misc    => "misc"
    case Adv(wh) => wh.toString
  }

  def fromString(str: String): Option[ArgumentSlot] = str match {
    case "subj"  => Some(Subj)
    case "obj"   => Some(Obj)
    case "prep1" | "prep1-obj" => Some(Prep1)
    case "prep2" | "prep2-obj" => Some(Prep2)
    case "misc"  => Some(Misc)
    case wh if allAdvSlots.contains(Adv(wh.lowerCase)) => Some(Adv(wh.lowerCase))
    case _ => None
  }

  import io.circe.{KeyEncoder, KeyDecoder}
  import io.circe.{Encoder, Decoder}
  import io.circe.Json

  implicit val argumentSlotEncoder: Encoder[ArgumentSlot] = Encoder[String].contramap(ArgumentSlot.toString)
  implicit val argumentSlotDecoder: Decoder[ArgumentSlot] = Decoder[String].emapTry(s => scala.util.Try(ArgumentSlot.fromString(s).get))

  implicit val argumentSlotKeyEncoder = KeyEncoder.instance(ArgumentSlot.toString)
  implicit val argumentSlotKeyDecoder = KeyDecoder.instance(ArgumentSlot.fromString)

  import cats.Id

  implicit val dependentArgumentEncoder = new DependentEncoder[ArgumentSlot.Aux, Id] {
    final def apply[A](slot: ArgumentSlot.Aux[A]) = slot match {
      case Adv(_) => implicitly[Encoder[Unit]].asInstanceOf[Encoder[A]] // TODO this should be acceptable w/o cast?
      case Subj   => implicitly[Encoder[Noun]]
      case Obj    => implicitly[Encoder[Noun]]
      case Prep1  => implicitly[Encoder[Preposition]]
      case Prep2  => implicitly[Encoder[Preposition]]
      case Misc   => implicitly[Encoder[NonPrepArgument]]
    }
  }

  implicit val dependentArgumentDecoder = new DependentDecoder[ArgumentSlot.Aux, Id] {
    final def apply[A](slot: ArgumentSlot.Aux[A]) = slot match {
      case Adv(_) => implicitly[Decoder[Unit]].asInstanceOf[Decoder[A]] // TODO this should be acceptable w/o cast?
      case Subj   => implicitly[Decoder[Noun]]
      case Obj    => implicitly[Decoder[Noun]]
      case Prep1  => implicitly[Decoder[Preposition]]
      case Prep2  => implicitly[Decoder[Preposition]]
      case Misc   => implicitly[Decoder[NonPrepArgument]]
    }
  }
}
