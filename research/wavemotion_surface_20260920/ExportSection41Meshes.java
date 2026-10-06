import com.comsol.model.*;
import com.comsol.model.util.*;

/** Read-only exports of the five actual h=20 um MCST mesh labels. */
public class ExportSection41Meshes {
  public static void main(String[] args) throws Exception {
    String root = System.getenv("MCST_PROJECT_ROOT");
    if (root == null || root.isEmpty()) root = ".";
    String run = System.getenv("COMSOL_MESH_RUN");
    if (run == null || run.isEmpty()) run = new java.io.File(root,
        "research/wavemotion_surface_20260920/runs/section41_mesh_20260923T125509Z").getPath();
    String assets = System.getenv("COMSOL_MESH_ASSETS");
    if (assets == null || assets.isEmpty()) assets = new java.io.File(root,
        "output/pdf/modal_selectivity_assets").getPath();
    run = run + java.io.File.separator;
    assets = assets + java.io.File.separator;
    String[] labels = {"M3_L4","M2_L2","M2_L4","M2_L6","M1_L4"};
    for (String label : labels) {
      if (!new java.io.File(run+"mcst_"+label+"/model_Model.mph").isFile())
        throw new java.io.FileNotFoundException("Historical MPH meshes are omitted from the public release; set COMSOL_MESH_RUN to the complete local mesh run.");
    }
    java.io.File assetDirectory = new java.io.File(assets);
    if (!assetDirectory.isDirectory() && !assetDirectory.mkdirs())
      throw new java.io.IOException("Could not create COMSOL_MESH_ASSETS output directory");
    for (String label : labels) {
      Model m = ModelUtil.load("Section41Mesh",run+"mcst_"+label+"/model_Model.mph");
      m.result().dataset().create("paperMesh","Mesh");
      m.result().dataset("paperMesh").set("mesh","mesh1");
      m.result().dataset("paperMesh").set("sorder","quadratic");
      m.result().create("paperMeshPlot","PlotGroup3D");
      m.result("paperMeshPlot").set("data","paperMesh");
      m.result("paperMeshPlot").set("titletype","none");
      m.result("paperMeshPlot").set("showlegends",false);
      m.result("paperMeshPlot").create("wire","Mesh");
      m.result("paperMeshPlot").feature("wire").set("elemcolor","type");
      m.result("paperMeshPlot").feature("wire").set("colorlegend","off");
      m.result("paperMeshPlot").feature("wire").set("wireframecolor","black");
      m.result("paperMeshPlot").feature("wire").set("resolution","norefine");
      m.component("comp1").view("view1").camera().set("projection","orthographic");
      m.component("comp1").view("view1").camera().set("position",new double[]{900,-1150,850});
      m.component("comp1").view("view1").camera().set("target",new double[]{0,0,0});
      m.component("comp1").view("view1").camera().set("up",new double[]{0,0,1});
      m.result("paperMeshPlot").run();
      m.result().export().create("paperMeshImage","paperMeshPlot","Image3D");
      m.result().export("paperMeshImage").set("imagetype","png");
      m.result().export("paperMeshImage").set("pngfilename",
          assets+"section41_mesh_"+label+".png");
      m.result().export("paperMeshImage").set("size","manualweb");
      m.result().export("paperMeshImage").set("width",1800);
      m.result().export("paperMeshImage").set("height",1050);
      m.result().export("paperMeshImage").set("resolution",220);
      m.result().export("paperMeshImage").set("antialias","on");
      m.result().export("paperMeshImage").set("background","color");
      m.result().export("paperMeshImage").set("customcolor",new double[]{1,1,1});
      m.result().export("paperMeshImage").set("legend3d","off");
      m.result().export("paperMeshImage").set("logo3d","off");
      m.result().export("paperMeshImage").set("grid","off");
      m.result().export("paperMeshImage").set("axisorientation","off");
      m.result().export("paperMeshImage").run();
      System.out.println("EXPORTED_ACTUAL_SECTION41_"+label);
      ModelUtil.clear();
    }
  }
}
