import qasrl.clause.ext.demo._
import qasrl.crowd._
import spacro._
import spacro.tasks._
import spacro.util._
import akka.pattern.ask
import scala.concurrent.duration._

import com.amazonaws.services.mturk._
import com.amazonaws.services.mturk.model._

import cats.implicits._

import jjm.ling.Text
import jjm.implicits._

import java.nio.file.Paths

val label = "trial"

val isProduction = false // sandbox. change to true for production
val domain = "recycle.cs.washington.edu" // change to your domain, or keep localhost for testing
val projectName = "qasrl-crowd-example" // make sure it matches the SBT project;
// this is how the .js file is found to send to the server

val interface = "0.0.0.0"
val httpPort = 7767
val httpsPort = 8080

val annotationPath = Paths.get(s"data/$label/annotations")
implicit val timeout = akka.util.Timeout(5.seconds)
implicit val config: TaskConfig = {
  if(isProduction) {
    val hitDataService = new FileSystemHITDataService(annotationPath.resolve("production"))
    ProductionTaskConfig(projectName, domain, interface, httpPort, httpsPort, hitDataService)
  } else {
    val hitDataService = new FileSystemHITDataService(annotationPath.resolve("sandbox"))
    SandboxTaskConfig(projectName, domain, interface, httpPort, httpsPort, hitDataService)
  }
}

def exit = {
  // actor system has to be terminated for JVM to be able to terminate properly upon :q
  config.actorSystem.terminate
  // flush & release logging resources
  import org.slf4j.LoggerFactory
  import ch.qos.logback.classic.LoggerContext
  LoggerFactory.getILoggerFactory.asInstanceOf[LoggerContext].stop
  System.out.println("Terminated actor system and logging. Type :q to end.")
}

val setup = new ClausalExample(label, Paths.get("data/clausal-predictions.jsonl"))

val exp = setup.experiment
exp.server

// use with caution... intended mainly for sandbox
def deleteAll = {
  exp.setValHITsActive(0)
  Thread.sleep(200)
  exp.expire
  exp.delete
}

def yesterday = {
  val cal = java.util.Calendar.getInstance
  cal.add(java.util.Calendar.DATE, -1)
  cal.getTime
}

import scala.collection.JavaConverters._

def expireHITById(hitId: String) = {
  config.service.updateExpirationForHIT(
    (new UpdateExpirationForHITRequest)
      .withHITId(hitId)
      .withExpireAt(yesterday))
}

def approveAllAssignmentsByHITId(hitId: String) = for {
  mTurkAssignment <- config.service.listAssignmentsForHIT(
    new ListAssignmentsForHITRequest()
      .withHITId(hitId)
      .withAssignmentStatuses(AssignmentStatus.Submitted)
    ).getAssignments.asScala.toList
} yield config.service.approveAssignment(
  new ApproveAssignmentRequest()
    .withAssignmentId(mTurkAssignment.getAssignmentId)
    .withRequesterFeedback(""))

def deleteHITById(hitId: String) =
  config.service.deleteHIT((new DeleteHITRequest).withHITId(hitId))

def disableHITById(hitId: String) = {
  expireHITById(hitId)
  deleteHITById(hitId)
}

def getActiveHITIds = {
  config.service.listAllHITs.map(_.getHITId)
}
