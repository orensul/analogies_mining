package qfirst.model.eval

import cats.Show
import cats.implicits._
import cats.data.NonEmptyList
import cats.effect.concurrent.Ref
import cats.effect.{ExitCode, IO, IOApp, Resource}

import com.monovore.decline._
import com.monovore.decline.effect._

import java.nio.file.{Path => NIOPath}
import java.nio.file.Files

import jjm.LowerCaseString
import jjm.ling.ESpan
import jjm.ling.Text
import jjm.ling.en.VerbForm
import jjm.metrics._
import jjm.implicits._
import jjm.io.FileUtil

import qasrl.bank.Data
import qasrl.bank.SentenceId

import qasrl.data.Dataset
import qasrl.data.VerbEntry
import qasrl.data.QuestionLabel

import qasrl.labeling.SlotBasedLabel

import fs2.Stream

import io.circe.{Encoder, Decoder}

object MetricsApp extends CommandIOApp(
  name = "qfirst.jvm.runMetrics",
  header = "Calculate QA-SRL metrics.") {

  import ErrorAnalysis._

  import Instances.Bucketers

  object MoreBucketers {
    val questionBucketers = Map(
      "wh" -> Bucketers.wh,
      "prep" -> Bucketers.prep
    )

    val templateBucketers = Map(
      "wh" -> Bucketers.Templated.wh,
      "prep" -> Bucketers.Templated.prep
    )

    val alignedSpanBucketers = Map(
      "gold-dep-len" -> Bucketers.goldDepLength(NonEmptyList.of(1, 2, 3, 5, 8, 12, 18, 27)),
      // "pred-dep-len" -> Bucketers.predDepLength(NonEmptyList.of(1, 2, 3, 5, 8, 12, 18, 27))
      )

    def nullBucketer[I] = Map.empty[String, I => String]

    def verbBucketers[A](verbFreq: (LowerCaseString => Int)) = Map(
      "verb-freq" -> Bucketers.verbFreq[A](
        verbFreq,
        NonEmptyList.of(0, 10, 50, 150, 250, 500, 750, 1000))
    )

    def sentenceBucketers[A] = Map(
      "sent-length" -> Bucketers.sentenceLength(NonEmptyList.of(0, 8, 16, 24, 32)).lmap[Instances.VerbInstance[A]](_.goldSentence)
    )

    def domainBucketers[A] = Map(
      "domain" -> ((verb: Instances.VerbInstance[A]) => SentenceId.fromString(verb.goldSentence.sentenceId).documentId.domain.toString)
    )
  }
  import MoreBucketers._

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

  object Rendering {

    // def renderQASetExample(qas: Instance.QASetInstance): String = {
    //   val verb = qas.goldVerb
    //   val sid = qas.goldSentence.sentenceId
    //   val sentenceTokens = qas.goldSentence.sentenceTokens
    //   val verbString = verb.verbInflectedForms.stem.toString + " (" + verb.verbIndex + ")"
    //   val allQAStrings = qas.allQuestionStrings.toList
    //     .filter(qString => renderInvalidGold || qas.goldValid.contains(qString) || qas.pred.contains(qString))
    //     .sortBy { qString =>
    //       if(qas.goldValid.contains(qString) && qas.pred.contains(qString)) -1
    //       else if(qas.goldValid.contains(qString)) 0
    //       else if(qas.goldInvalid.contains(qString) && qas.pred.contains(qString)) 1
    //       else if(qas.pred.contains(qString)) 2
    //       else 3 }
    //     .map { qString =>
    //       val isGoldInvalid = qas.goldInvalid.contains(qString)
    //       val renderedQ = (if(isGoldInvalid) "#" else "") + qString
    //       val goldString = qas.goldValid.get(qString).orElse(
    //         qas.goldInvalid.get(qString).filter(_ => renderInvalidGold)
    //       ).fold("\t\t") { qLabel =>
    //         val spans = qLabel.answerJudgments.toList.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet[ESpan].toList.sortBy(_.begin)
    //         val renderedSpans = spans.map(Text.renderSpan(sentenceTokens, _)).mkString(" / ")
    //         renderedQ + "\t" + renderedSpans + "\t"
    //       }
    //       val predString = qas.pred.get(qString).fold("\t") { case (_, spans) =>
    //         val renderedSpans = spans.map(Text.renderSpan(sentenceTokens, _)).mkString(" / ")
    //         renderedQ + "\t" + renderedSpans
    //       }
    //       goldString + predString
    //     }
    //   "\t" + Text.render(sentenceTokens) + "\n" +
    //     "\t" + verbString + "\n" +
    //     allQAStrings.map("\t" + _).mkString("\n")
    // }

    def renderQuestionExample(question: Instances.QuestionInstance, renderInvalidGold: Boolean = false): String = {
      val qas = question.qas
      val verb = qas.goldVerb
      val sid = qas.goldSentence.sentenceId
      val sentenceTokens = qas.goldSentence.sentenceTokens
      val verbString = verb.verbInflectedForms.stem.toString + " (" + verb.verbIndex + ")"
      val allQAStrings = qas.allQuestionStrings.toList
        .filter(qString => renderInvalidGold || qas.goldValid.contains(qString) || qas.pred.contains(qString))
        .sortBy { qString =>
          if(qas.goldValid.contains(qString) && qas.pred.contains(qString)) -1
          else if(qas.goldValid.contains(qString)) 0
          else if(qas.goldInvalid.contains(qString) && qas.pred.contains(qString)) 1
          else if(qas.pred.contains(qString)) 2
          else 3 }
        .map { qString =>
          val isGoldInvalid = qas.goldInvalid.contains(qString)
          val renderedQ = (if(qString == question.string) "*" else "") + (if(isGoldInvalid) "#" else "") + qString
          val goldString = qas.goldValid.get(qString).orElse(
            qas.goldInvalid.get(qString).filter(_ => renderInvalidGold)
          ).fold("\t\t") { qLabel =>
            val spans = qLabel.answerJudgments.toList.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet[ESpan].toList.sortBy(_.begin)
            val renderedSpans = spans.map(Text.renderSpan(sentenceTokens, _)).mkString(" / ")
            renderedQ + "\t" + renderedSpans + "\t"
          }
          val predString = qas.pred.get(qString).fold("\t") { case (_, spans) =>
            val renderedSpans = spans.map(Text.renderSpan(sentenceTokens, _)).mkString(" / ")
            renderedQ + "\t" + renderedSpans
          }
          goldString + predString
        }
      "\t" + Text.render(sentenceTokens) + "\n" +
        "\t" + verbString + "\n" +
        allQAStrings.map("\t" + _).mkString("\n")
    }
    val renderQASetExample = (si: Instances.QASetInstance) => {
      Text.render(si.goldSentence.sentenceTokens) + "\n" + {
        val verbStr = s"${si.goldVerb.verbInflectedForms.stem} (${si.goldVerb.verbIndex})"
        val goldValidStr = si.goldValid.toList.map {
          case (qString, qLabel) =>
            val answerSpans = qLabel.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet
            val spanStr = answerSpans.toList.sortBy(_.begin).map(s =>
              Text.renderSpan(si.goldSentence.sentenceTokens, s)
            ).mkString(" / ")
            f"$qString%-60s $spanStr%s"
        }.mkString("\n")
        val goldInvalidStr = si.goldInvalid.toList.map {
          case (qString, qLabel) =>
            val answerSpans = qLabel.answerJudgments.flatMap(_.judgment.getAnswer).flatMap(_.spans.toList).toSet
            val spanStr = answerSpans.toList.sortBy(_.begin).map(s =>
              Text.renderSpan(si.goldSentence.sentenceTokens, s)
            ).mkString(" / ")
            f"$qString%-60s $spanStr%s"
        }.mkString("\n")
        val predStr = si.pred.toList.map {
          case (qString, (qSlots, answerSpans)) =>
            val answerSpanStr = answerSpans.toList.sortBy(_.begin).map(s =>
              Text.renderSpan(si.goldSentence.sentenceTokens, s)
            ).mkString(" / ")
            f"$qString%-60s $answerSpanStr%s"
        }.mkString("\n")
        verbStr + "\nGOLD:\n" + goldValidStr + "\nGOLD INVALID:\n" + goldInvalidStr + "\nPREDICTED:\n" + predStr
      }
    }
  }
  import Rendering._

  val I = Instances
  import jjm.metrics.{Transformers => M}
  import shapeless._
  import shapeless.syntax.singleton._
  import shapeless.record._
  import monocle.function.{all => Optics}

  def constructQASetInstances[Beam, Filter, FilterSpace](
    protocol: BeamProtocol[Beam, Filter, FilterSpace])(
    gold: Dataset,
    filterGold: VerbEntry => (Map[String, QuestionLabel], Map[String, QuestionLabel]),
    filterPred: Filter,
  ) = (
    pred: SentencePrediction[Beam]
  ) => {
    val goldSentence = gold.sentences(pred.sentenceId)
    pred.verbs.map { predVerb =>
      val goldVerb = goldSentence.verbEntries(predVerb.verbIndex)
      val (goldInvalidQAs, goldValidQAs) = filterGold(goldVerb)
      val predQAs = protocol.filterBeam(filterPred, predVerb)
      I.QASetInstance(goldSentence, goldVerb, goldValidQAs, goldInvalidQAs, predQAs)
    }: List[I.QASetInstance]
  }

  val computeQASetMetrics = M.split(I.qaSetToQuestions) {
    M.hchoose(
      "question" ->> I.getQuestionBoundedAcc,
      "question with answer" ->> I.getQuestionWithAnswerBoundedAcc,
      "question-answer pair" ->> M.split(I.questionToQAs) {
        I.getQABoundedAcc
      }
    )
  }

  val computeQASetAnalysisMetrics = (qas: I.QASetInstance) => {
    val gen = qas.generalize
    val genTemplated = gen.map(TemplateSlots.fromQuestionSlots)
    "original" ->> gen.getAcc ::
    "no tense/aspect/modality" ->> gen.map(I.stripTense).getAcc ::
      "no negation" ->> gen.map(I.stripNegation).getAcc ::
      "no tense/aspect/modality/negation" ->> gen.map(I.stripNegation).map(I.stripTense).getAcc ::
      "no animacy" ->> gen.map(I.stripAnimacy).getAcc ::
      "templated" ->> genTemplated.getAcc ::
      "templated: no answers" ->> genTemplated.stripAnswers.getAcc ::
      "templated: no wh" ->> genTemplated.map(I.stripTemplateWh).getAcc ::
      "templated: no prep" ->> genTemplated.map(I.stripTemplatePrep).getAcc ::
      "templated: nothing" ->> genTemplated.stripAnswers.map(I.stripTemplateAll).getAcc ::
      HNil
  }

  def sentenceDomainBucketers[A] = Map(
    "domain" -> ((pred: SentencePrediction[A]) => SentenceId.fromString(pred.sentenceId).documentId.domain.toString)
  )

  def computeSentenceMetricsForAllFilters[Beam, Filter, FilterSpace](
    protocol: BeamProtocol[Beam, Filter, FilterSpace])(
    gold: Dataset,
    filterSpace: FilterSpace,
  ) = {
    M.bucket(sentenceDomainBucketers[Beam]) {
      M.hchoose(
        "verbs" ->> M.split(((s: SentencePrediction[Beam]) => s.verbs)) {
          (v: VerbPrediction[Beam]) => Count(v.verbInflectedForms)
        },
        "predictions" ->> M.choose(protocol.getAllFilters(filterSpace)) { filter =>
          M.split(constructQASetInstances(protocol)(gold, filterGoldDense, filter)) {
            computeQASetMetrics
          }
        }
      )
    }
  }

  def computePredClassesForSentences[Beam, Filter, FilterSpace](
    protocol: BeamProtocol[Beam, Filter, FilterSpace])(
    gold: Dataset,
    filter: Filter,
  ) = {
    M.bucket(sentenceDomainBucketers[Beam]) {
      M.split(constructQASetInstances(protocol)(gold, filterGoldDense, filter)) {
        M.split(I.qaSetToQuestions) {
          ((q: I.QuestionInstance) => computePredClass(q) -> q) andThen Count[(PredClass, I.QuestionInstance)]
          // (computePredClass *** identity[I.QuestionInstance]) andThen Count[(PredClass, I.QuestionInstance)]
        }
      }
    }
  }

  def writeExamples(path: NIOPath, examples: Vector[Instances.QuestionInstance], rand: util.Random) = {
    val examplesString = rand.shuffle(examples.map(renderQuestionExample(_))).mkString("\n")
    IO(Files.write(path, examplesString.getBytes("UTF-8")))
  }

  def getMetricsString[M: HasMetrics](m: M) =
    m.getMetrics.toStringPrettySorted(identity, x => x.render, sortSpec)

  def runDenseMetrics[Beam, Filter: Show, FilterSpace: Encoder : Decoder](
    protocol: BeamProtocol[Beam, Filter, FilterSpace])(
    gold: Dataset,
    predStream: Stream[IO, SentencePrediction[Beam]],
    metadataDir: NIOPath,
    recomputeFilter: Boolean,
    isTest: Boolean
  ) = {
    val filtersPath = metadataDir.resolve("filters.json")
    val filterSpaceIO = FileUtil.readJson[FilterSpace](filtersPath)
      .flatMap { fs =>
        if(isTest) {
          if(protocol.getAllFilters(fs).size > 1) {
            IO(throw new RuntimeException("Must run on test with only a single filter."))
          } else IO.pure(fs)
        } else if(recomputeFilter) {
          IO.pure(protocol.withBestFilter(fs, None))
        } else IO.pure(fs)
      }

    for {
      filterSpace <- filterSpaceIO
      raw <- predStream.map(
        computeSentenceMetricsForAllFilters(protocol)(gold, filterSpace)
      ).compile.foldMonoid
      rawCollapsed = raw.collapsed
      tunedResults = {
        // compute number of questions and spans per verb
        val allResults = rawCollapsed.updateWith("predictions")(
          _.map { stats => // stats for given filter
            val questionsPerVerb = "questions per verb" ->> stats.get("question").stats.predicted.toDouble / rawCollapsed("verbs").stats.numInstances
            val spansPerVerb = "spans per verb" ->> stats.get("question-answer pair").stats.predicted.toDouble / rawCollapsed("verbs").stats.numInstances
            stats + questionsPerVerb + spansPerVerb
          }
        )
        // Recall restriction: choose the filter with best results that has >= 2 Qs/verb and >= 2.3 spans/verb
        val qRecallRestriction = if(isTest) 0.0 else 2.0
        val sRecallRestriction = if(isTest) 0.0 else 2.3
        allResults.updateWith("predictions")(
          _.filter(_.get("questions per verb") >= qRecallRestriction)
            .filter(_.get("spans per verb") >= sRecallRestriction)
            .keepMaxBy(_.get("question-answer pair").stats.accuracyLowerBound)
        )
      }
      bestFilter <- IO {
        tunedResults.get("predictions").data.headOption match {
          case Some(f) => f._1
          case None => throw new RuntimeException("No filter in the specified space satisfies the recall threshold.")
        }
      }
      _ <- {
        predStream.take(5)
          .flatMap(p => Stream.emits(constructQASetInstances(protocol)(gold, filterGoldDense, bestFilter)(p)))
          .map(renderQASetExample)
          .flatMap(s => Stream.eval(IO(println(s))))
          .compile.drain
      }

      _ <- {
        if(!isTest) {
          import io.circe.generic.auto._
          FileUtil.writeJson(filtersPath, io.circe.Printer.spaces4)(protocol.withBestFilter(filterSpace, Some(bestFilter)))
        } else IO.unit
      }

      _ <- IO(println("Tuned results: " + getMetricsString(tunedResults)))

      analysisResults <- {
        predStream
          .flatMap(p => Stream.emits(constructQASetInstances(protocol)(gold, filterGoldDense, bestFilter)(p)))
          .map(computeQASetAnalysisMetrics)
          .compile.foldMonoid
      }

      _ <- IO(println("Analysis results: " + getMetricsString(analysisResults)))

      examplesDir <- IO {
        val examplesDirectory = metadataDir.resolve("examples")
        if(!Files.exists(examplesDirectory)) {
          Files.createDirectories(examplesDirectory)
        }
        examplesDirectory
      }
      _ <- IO {
        val otherIncorrect = analysisResults("templated: nothing").incorrect
        def renderGenQuestionExample(
          instance: Instances.GeneralizedQASetInstance[TemplateSlots],
          question: TemplateSlots
        ) = {

        }
        def writeGenExamples(
          path: NIOPath,
          examples: Vector[(Instances.GeneralizedQASetInstance[TemplateSlots], TemplateSlots, ESpan)],
          rand: util.Random
        ) = {
          val exampleStrings = examples.map { case (instance, templateQ, _) =>
            renderQASetExample(instance.original) + "\n" + templateQ.toTemplateString + "\n" + templateQ + "\n\n"
          }
          val examplesString = rand.shuffle(exampleStrings).mkString("\n")
          IO(Files.write(path, examplesString.getBytes("UTF-8")))
        }

        val r = new util.Random(235867962L)
        writeGenExamples(examplesDir.resolve("full-other.txt"), otherIncorrect, r)
      }.flatten

      // TODO: here is potentially useful code for pulling out examples specific to an error case for printing later.
      // _ <- IO {
      //   val correctos = analysisResults("no negation").correct.groupBy(x => x._1.original -> x._3)
      //   val incorrectos = analysisResults("original").incorrect.groupBy(x => x._1.original -> x._3)
      //   val negationFixos = incorrectos.keySet.intersect(correctos.keySet)
      //   negationFixos.foreach(fixo =>
      //     println(renderQASetExample(fixo._1) + "\n\n")
      //   )
      // }

      // _ <- IO {
      //   println("\n\n== WRONGO, WRONGO, BIG CHEESE ==\n\n")
      //   val correctos = analysisResults("no negation").incorrect.groupBy(x => x._1.original -> x._3)
      //   val incorrectos = analysisResults("original").correct.groupBy(x => x._1.original -> x._3)
      //   val negationWrongos = incorrectos.keySet.intersect(correctos.keySet)
      //   negationWrongos.foreach(wrongo =>
      //     println(renderQASetExample(wrongo._1) + "\n\n")
      //   )
      // }

      // performance on verbs by frequency

      // val verbBucketedResults = raw.map(
      //   _.updateWith("predictions")(
      //     _.data(bestFilter).get("question with answer")
      //   )
      // )
      // println("Overall by verb: " + getMetricsString(verbBucketedResults))

      // predClasses <- predStream.map(
      //   computePredClassesForSentences(protocol)(gold, bestFilter)
      // ).map(_.map(_.filter(_._1 != PredClass.NotPredicted))).compile.foldMonoid

      // mainErrorClasses = {
      //   import PredClass._
      //   predClasses.map(
      //     _.values.map(_._1).filter(_ != PredClass.Correct).map {
      //       case WrongWh(_, _) => WrongWh("_".lowerCase, "_".lowerCase)
      //       case SwappedPrep(_, _) => SwappedPrep("_".lowerCase, "_".lowerCase)
      //       case MissingPrep(_) => MissingPrep("_".lowerCase)
      //       case ExtraPrep(_) => ExtraPrep("_".lowerCase)
      //       case x => x
      //     }
      //   )
      // }

      // bucketedErrorClasses = mainErrorClasses.map(
      //   _.foldMap(
      //     M.bucket(Map("class" -> ((x: PredClass) => x.toString))) {
      //       ((_: PredClass) => 1)
      //     }
      //   )
      // )

      // whConf = {
      //   import PredClass._
      //   predClasses.collapsed.values.collect {
      //     case (Correct | WrongAnswer | CorrectTemplate, q) => Confusion.instance(q.slots.wh, q.slots.wh, q)
      //     case (WrongWh(pred, gold), q) => Confusion.instance(gold, pred, q)
      //   }.combineAll
      // }

      // _ <- IO {
      //   println(whConf.stats.prettyString(0))
      //   println("Collapsed error classes: " + getMetricsString(bucketedErrorClasses.collapsed))
      //   // println("All bucketed error classes: " + getMetricsString(bucketedErrorClasses))
      // }

      // collapsedPredClasses = predClasses.collapsed

      // examplesDir = metadataDir.resolve("examples")
      // _ <- IO {
      //     if(!Files.exists(examplesDir)) {
      //       Files.createDirectories(examplesDir)
      //     }
      //     val r = new util.Random(235867962L)
      //     writeExamples(
      //       examplesDir.resolve("template.txt"),
      //       collapsedPredClasses.values.collect { case (PredClass.CorrectTemplate, q) => q },
      //       r
      //     ) >> writeExamples(
      //       examplesDir.resolve("prep-swap.txt"),
      //       collapsedPredClasses.values.collect { case (PredClass.SwappedPrep(_, _), q) => q },
      //       r
      //     ) >> writeExamples(
      //       examplesDir.resolve("other.txt"),
      //       collapsedPredClasses.values.collect { case (PredClass.Other, q) => q },
      //       r
      //     ) >> writeExamples(
      //       examplesDir.resolve("wrongans.txt"),
      //       collapsedPredClasses.values.collect { case (PredClass.WrongAnswer, q) => q },
      //       r
      //     )
      // }.flatten

      // _ <- IO {
      //   val domainsDir = examplesDir.resolve("domain")
      //   if(!Files.exists(domainsDir)) {
      //     Files.createDirectories(domainsDir)
      //   }
      //   {
      //     val r = new util.Random(22646L)
      //     predClasses.data.foreach { case (buckets, results) =>
      //       // val instances = results.get("predictions").incorrect ++ results.get("predictions").uncertain
      //       val bucketDir = domainsDir.resolve(buckets("domain"))
      //       if(!Files.exists(bucketDir)) {
      //         Files.createDirectories(bucketDir)
      //       }
      //       val instances = results.values.collect { case (PredClass.Other, q) => q }
      //       writeExamples(bucketDir.resolve("other.txt"), instances, r)
      //       val ansInstances = results.values.collect { case (PredClass.WrongAnswer, q) => q }
      //       writeExamples(bucketDir.resolve("wrongans.txt"), ansInstances, r)
      //     }
      //   }
      // }
    } yield ()
  }

  import qfirst.clause.ext._
  import qfirst.clause.ext.ClauseDataWriter.ClauseInfo

  def getClauseMapping(
    clauseInfo: Map[String, Map[Int, Map[String, ClauseInfo]]]
  ): Map[String, ArgStructure] = for {
    (_, verbToQuestions) <- clauseInfo
    (_, questionToFrame) <- verbToQuestions
    (_, clauseInfo) <- questionToFrame
  } yield {
    val structure = clauseInfo.frame.structure.forgetAnimacy
    def getArgStr[A <: qfirst.clause.ext.Argument](arg: ArgumentSlot.Aux[A]) = {
      structure.args.get(arg).fold("_")(_.placeholder.mkString(" "))
    }
    val frameString = List(
      getArgStr(Subj),
      (if(structure.isPassive) "verb[pss]" else "verb"),
      getArgStr(Obj),
      structure.args.get(Prep1).fold("_")(_.preposition.toString),
      structure.args.get(Prep1).flatMap(_.objOpt).fold("_")(_.placeholder.mkString(" ")),
      structure.args.get(Prep2).fold("_")(_.preposition.toString),
      structure.args.get(Prep2).flatMap(_.objOpt).fold("_")(_.placeholder.mkString(" ")),
      getArgStr(Misc)
    ).mkString(" ")
    frameString -> structure
  }

  def getVerbFrequencies(data: Dataset) = {
    data.sentences.iterator
      .flatMap(s => s._2.verbEntries.values.map(_.verbInflectedForms.stem).iterator)
      .foldLeft(Map.empty[LowerCaseString, Int].withDefaultValue(0)) {
      (counts, stem) => counts + (stem -> (counts(stem) + 1))
    }
  }

  def readClauseInfo(
    path: NIOPath
  ): IO[Map[String, Map[Int, Map[String, ClauseInfo]]]] = {
    FileUtil.readJsonLines[ClauseInfo](path)
      .map(fi => Map(fi.sentenceId -> Map(fi.verbIndex -> Map(fi.question -> List(fi)))))
      .compile.foldMonoid.map(
        // non-ideal.. would instead want a recursive combine that overrides and doesn't need the end mapping
        _.transform { case (sid, vs) => vs.transform { case (vi, qs) => qs.transform { case (q, fis) => fis.head } } }
      )
  }

  sealed trait MetricsMode {
    override def toString = this match {
      case Dense => "dense"
      case DenseCurve => "dense-curve"
    }
  }
  case object Dense extends MetricsMode
  case object DenseCurve extends MetricsMode

  def programInternal[Beam: Decoder, Filter: Show, FilterSpace: Encoder : Decoder](
    protocol: BeamProtocol[Beam, Filter, FilterSpace])(
    verbFrequencies: IO[LowerCaseString => Int], devDense: Dataset,
    predDir: NIOPath,
    clauseInfoOpt: Option[IO[Map[String, Map[Int, Map[String, ClauseInfo]]]]],
    mode: MetricsMode, recomputeFilter: Boolean,
    isTest: Boolean
  ) = {
    val predFile = if(isTest) {
      predDir.resolve("predictions-test.jsonl")
    } else predDir.resolve("predictions.jsonl")
    val metadataDir = predDir.resolve(mode.toString)
    clauseInfoOpt match {
      case Some(clauseInfoIO) => // TODO for e2e model / eval
        // for {
        //   predictions = FileUtil.streamE2EPredictions(getClauseMapping(clauseInfo), predFile)
        //   _ <- runE2EDenseMetrics(verbFrequencies, gold, predictions, metadataDir, recomputeFilter)
        // } yield ()
        IO(())
      case None =>
        val pred = FileUtil.readJsonLines[SentencePrediction[Beam]](predFile)
        mode match {
          case Dense => runDenseMetrics(protocol)(devDense, pred, metadataDir, recomputeFilter, isTest)
          case DenseCurve => IO(()) // TODO
        }
    }
  }

  def program(
    qasrlBankPath: NIOPath, predDir: NIOPath,
    clauseInfoPathOpt: Option[NIOPath],
    mode: String, protocolName: String, recomputeFilter: Boolean,
    isTest: Boolean
  ): IO[ExitCode] = for {
    metricsMode <- IO {
      mode match {
        case "dense" => Dense
        case "dense-curve" => DenseCurve
        case _ => throw new RuntimeException("Must specify mode of dense-curve or dense.")
      }
    }
    trainIO = IO.fromTry(qasrl.bank.Data.readQasrlDataset(qasrlBankPath.resolve("orig").resolve("train.jsonl.gz")))
    verbFrequenciesIO = trainIO.map(getVerbFrequencies)
    dataset <- (
      if(isTest) {
        IO.fromTry(qasrl.bank.Data.readQasrlDataset(qasrlBankPath.resolve("dense").resolve("test.jsonl.gz")))
      } else {
        IO.fromTry(qasrl.bank.Data.readQasrlDataset(qasrlBankPath.resolve("dense").resolve("dev.jsonl.gz")))
      }
    )
    clauseInfoOpt = clauseInfoPathOpt.map(readClauseInfo)
    beamProtocol <- {
      import qfirst.model.eval.protocols._
      import io.circe.generic.auto._
      def go[Beam: Decoder, Filter: Show, FilterSpace: Encoder : Decoder](
        protocol: BeamProtocol[Beam, Filter, FilterSpace]
      ) = programInternal(protocol)(
        verbFrequenciesIO, dataset,
        predDir, clauseInfoOpt,
        metricsMode, recomputeFilter,
        isTest)
      protocolName match {
        case "afirst" => go(SimpleQAs.protocol[SlotBasedLabel[VerbForm]]())
        case "afirst-maxq" => go(SimpleQAs.protocol[SlotBasedLabel[VerbForm]](true))
        case "afirst-clausal" => go(SimpleQAs.protocol[Map[String, String]]())
        case "afirst-clausal-maxq" => go(SimpleQAs.protocol[Map[String, String]](true))
        case "qfirst-full" => go(QfirstFullProtocol)
        case "qfirst-no_tan" => go(QfirstNoTanProtocol)
        case "qfirst-clausal" => go(QfirstClausalProtocol)
        case "qfirst-clausal_no_tan" => go(QfirstClausalNoTanProtocol)
        case "qfirst-clausal_no_anim" => go(QfirstClausalNoAnimProtocol)
        case "qfirst-clausal_no_tan_or_anim" => go(QfirstClausalNoTanOrAnimProtocol)
        case "factored" => go(JointClauseSpanProtocol)
        case _ => throw new RuntimeException("Must specify a known beam protocol type.")
      }
    }
    // _ <- IO {
    //   import qfirst.model.eval.protocols.QfirstClausalNoTanOrAnimProtocol._
    //   println(f"Num questions predicted: $numQuestionsPredicted%s")
    //   println(f"Num questions edited: $numQuestionsEdited (${numQuestionsEdited * 100.0 / numQuestionsPredicted}%.2f%%)")
    //   println(f"Num edited questions produced: $numEditedQuestionsProduced (${numEditedQuestionsProduced / numQuestionsEdited}%.2f)")
    //   println(f"Num different questions produced: $numDifferentQuestionsProduced (${numDifferentQuestionsProduced * 100.0 / numEditedQuestionsProduced}%.2f%%)")
    // }
  } yield ExitCode.Success

  def main: Opts[IO[ExitCode]] = {
    val goldPath = Opts.option[NIOPath](
      "gold", metavar = "path", help = "Path to the QA-SRL Bank."
    )
    val predPath = Opts.option[NIOPath](
      "pred", metavar = "path", help = "Path to the directory of predictions."
    )
    val clauseInfoPathOpt = Opts.option[NIOPath](
      "clause-info", metavar = "path", help = "Path to the directory of clause resolutions of the data for running e2e metrics."
    ).orNone
    val mode = Opts.option[String](
      "mode", metavar = "non-dense|dense-curve|dense", help = "Which eval to run."
    ).withDefault("dense")
    val protocol = Opts.option[String](
      "protocol", metavar = "afirst|qfirst-{full,no_tan,...}|factored", help = "Which input type to handle."
    ).withDefault("qfirst-full")
    val recomputeFilter = Opts.flag(
      "recomputeFilter", help = "Whether to recompute the best filter as opposed to using the cached one."
    ).orFalse
    val isTest = Opts.flag(
      "test", help = "Whether run on test predictions."
    ).orFalse

    (goldPath, predPath, clauseInfoPathOpt, mode, protocol, recomputeFilter, isTest).mapN(program)
  }
}
