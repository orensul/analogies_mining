package qfirst.clause.ext

import jjm.LowerCaseString
import jjm.implicits._

import io.circe.generic.JsonCodec

import monocle.macros.Lenses
import monocle.macros.GenPrism

@JsonCodec sealed trait Argument {
  def placeholder: List[String]
  def gap: List[String]
  def unGap: List[String]
  def wh: Option[String]

  def isNoun: Boolean = this match {
    case Noun(_) => true
    case _       => false
  }

  def isPreposition: Boolean = this match {
    case Preposition(_, _) => true
    case _          => false
  }

  def isLocative: Boolean = this match {
    case Locative => true
    case _        => false
  }

  def isComplement: Boolean = this match {
    case Complement(_) => true
    case _ => false
  }
}

@JsonCodec @Lenses case class Preposition(
  preposition: LowerCaseString,
  objOpt: Option[NounLikeArgument]
) extends Argument {
  override def placeholder = objOpt.toList.flatMap(_.placeholder)
  override def gap = List(preposition.toString) ++ objOpt.toList.flatMap(_.gap)
  override def unGap = List(preposition.toString) ++ objOpt.toList.flatMap(_.unGap)
  override def wh = objOpt.flatMap(_.wh)
}
object Preposition {
  val isAnimate = Preposition.objOpt
    .composePrism(monocle.std.option.some)
    .composeOptional(NounLikeArgument.isAnimate)
}

@JsonCodec sealed trait NonPrepArgument extends Argument

case object Locative extends NonPrepArgument {
  override def placeholder = List("somewhere")
  override def gap = Nil
  override def unGap = Nil
  override def wh = Some("Where")

  def fromPlaceholder(s: LowerCaseString) = s.toString match {
    case "somewhere" => Some(Locative)
    case _ => None
  }
}

case class Complement(conj: Complement.Form) extends NonPrepArgument {
  override def placeholder = conj.placeholder
  override def gap = conj.gap
  override def unGap = conj.unGap
  override def wh = conj.wh
}
object Complement {
  @JsonCodec sealed trait Form {
    import Form._
    def placeholder = this match {
      case Infinitive => List("do", "something")
      case Bare => List("do", "something")
    }
    def gap = this match {
      case Infinitive => List("to", "do")
      case Bare => List("do")
    }
    def unGap = this match {
      case Infinitive => List("to")
      case Bare => List()
    }
    def wh = Some("What")
  }
  object Form {
    case object Infinitive extends Form
    case object Bare extends Form
  }

  def fromPlaceholder(s: LowerCaseString) = s.toString match {
    case "to do something" => Some(Complement(Form.Infinitive))
    case "do something" => Some(Complement(Form.Bare))
    case _ => None
  }
}

@JsonCodec sealed trait NounLikeArgument extends NonPrepArgument

case object Gerund extends NounLikeArgument {
  override def placeholder = List("doing", "something")
  override def gap = List("doing")
  override def unGap = List()
  override def wh = Some("What")

  def fromPlaceholder(s: LowerCaseString) = s.toString match {
    case "doing something" => Some(Gerund)
    case _ => None
  }
}

@JsonCodec @Lenses case class Noun(
  isAnimate: Boolean
) extends NounLikeArgument {
  override def placeholder = List(if (isAnimate) "someone" else "something")
  override def gap = Nil
  override def unGap = Nil
  override def wh = if (isAnimate) Some("Who") else Some("What")
}
object Noun {
  def fromPlaceholder(s: LowerCaseString) = s.toString match {
    case "someone"   => Some(Noun(isAnimate = true))
    case "something" => Some(Noun(isAnimate = false))
    case _ => None
  }
}

object NounLikeArgument {
  def fromPlaceholder(s: LowerCaseString) =
    Noun.fromPlaceholder(s).orElse(
      Gerund.fromPlaceholder(s)
    )

  val noun = GenPrism[NounLikeArgument, Noun]
  val gerund = GenPrism[NounLikeArgument, Gerund.type]

  val isAnimate = noun.composeLens(Noun.isAnimate)
}

object NonPrepArgument {
  def fromPlaceholder(s: LowerCaseString) =
    Noun.fromPlaceholder(s)
      .orElse(Gerund.fromPlaceholder(s))
      .orElse(Complement.fromPlaceholder(s))
      .orElse(Locative.fromPlaceholder(s))

  val nounLikeArgument = GenPrism[NonPrepArgument, NounLikeArgument]
  val isAnimate = nounLikeArgument.composeOptional(NounLikeArgument.isAnimate)
}
